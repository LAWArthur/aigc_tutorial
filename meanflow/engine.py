import argparse
import math
import gc
import os
import PIL
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

import numpy as np
from typing import Iterable, Callable, Any

from utils import distributed_utils
from utils.misc import MetricLogger, SmoothedValue
from utils.misc import print_rank_0
from data.augment import AugmentPipe
import rng


# Setup Augment Pipeline
augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)  # turn off yflip and rotate

# -------------- helpful training tools --------------
def synchronize_gradients(model: torch.nn.Module):
    """
    In a distributed setting, to enable jvp, we need to call model.module instead of model directly.
    If so, we synchronize gradients across all processes.
    """
    if not isinstance(model, DistributedDataParallel):
        return

    torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()

def gradient_sanity_check(model):
    if not isinstance(model, DistributedDataParallel):
        return
    torch.cuda.synchronize()
    # logging.info(f"Gradient sanity check ...")
    for name, p in model.module.named_parameters():
        if p.requires_grad and len(p.shape) > 3:
            monitor = p.grad.norm()

            monitor_list = [torch.zeros_like(monitor) for _ in range(dist.get_world_size())]
            dist.all_gather(monitor_list, monitor)
            monitor_tensor = torch.stack(monitor_list)
            # logging.info(f"All_gathered grad norm, param {name}: ")
            # for i, m in enumerate(monitor_tensor):
            #     logging.info(f"Rank {i}: {m:.16f}")
            # break

            # Assert all gradient norms are close to rank 0's
            ref = monitor_tensor[0]
            for i, m in enumerate(monitor_tensor):
                assert torch.isclose(m, ref), \
                    f"Gradient norm mismatch at rank {i}: {m} vs rank 0: {ref}"

def get_compiled_counts():
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return len(metrics)

def train_step(model_without_ddp, samples, class_labels, aug_cond=None):
    loss = model_without_ddp.forward_with_loss(x=samples, class_labels=class_labels, aug_cond=aug_cond)
    loss.backward(create_graph=False)
    return loss

# -------------- Train loop --------------
def train_one_epoch(
        args: argparse.Namespace,
        device: torch.device,
        model: torch.nn.Module,
        compiled_train_step: Callable,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        local_rank: int,
        tblogger: Any,
        ):
    # switch to train mode
    model.train(True)

    # declare the unwrapped model
    model_without_ddp = model if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module

    # setup metric logger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{} / {}]'.format(epoch, args.epochs)
    print_freq = args.log_per_step

    # train one epoch
    for iter_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        n_iter = iter_i + epoch * len(data_loader)

        # To device
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # scaling to [-1, 1] from [0, 1]
        samples = samples * 2.0 - 1.0

        # augment samples
        samples, aug_cond = rng.augment_with_rng_control(
            augment_pipe = augment_pipe,
            samples = samples,
            base_seed = args.seed,
            steps = n_iter,
            ) if args.use_edm_aug else (samples, None)

        if args.compile and epoch == args.start_epoch and iter_i == 0:
            print_rank_0(" > Compiling the first train step, this may take a while...")

        # prepare
        if args.class_cond:
            class_labels = F.one_hot(targets, num_classes=args.num_classes,).float()
        else:
            class_labels = None

        # calculate training losses
        loss = rng.train_step_with_rng_control(
            train_step_fn = compiled_train_step,
            model_without_ddp = model_without_ddp,
            step = n_iter,
            base_seed = args.seed,
            samples = samples,
            class_labels = class_labels,
            aug_cond = aug_cond,
            )
        if args.compile:
            assert get_compiled_counts() > 0, "Compilation not triggered."

        # sanity check
        synchronize_gradients(model)  # To support compiling, we need to call model.module and then sync gradients.
        if (epoch - args.start_epoch) % 100 == 0 and iter_i < 2:  # sanity check after the first steps
            gradient_sanity_check(model)

        # check losses
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # Backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model_without_ddp.update_ema()  # moved to begin of train_step
        optimizer.zero_grad()

        # adjust lr per-iteration
        lr_schedule.step()

        # updata metric logger
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)

        # updata tensorboard
        if tblogger is not None:
            epoch_1000x = int((iter_i / len(data_loader) + epoch) * 1000)
            tblogger.add_scalar("loss", loss_value, epoch_1000x)

        # debug mode
        if hasattr(args, "debug") and args.debug and iter_i > 2 * print_freq:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_rank_0("Averaged stats: {}".format(metric_logger), local_rank)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# -------------- Eval loop --------------
def eval_one_epoch(
    args: argparse.Namespace,
    model: DistributedDataParallel,
    model_ema: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    suffix: str = "",
):
    gc.collect()
    model.train(False)

    if args.distributed:
        data_loader.sampler.set_epoch(0)

    assert args.fid_samples <= len(data_loader.dataset), (
        f"In this interface, dataset size ({len(data_loader.dataset)}) must be larger than FID samples ({args.fid_samples})."
    )
    fid_samples = math.ceil(args.fid_samples / distributed_utils.get_world_size())

    # setup FID metric
    fid_metric = FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)

    num_synthetic = 0
    snapshots_saved = False
    if args.save_image_dir:
        (Path(args.save_image_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    # setup metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Eval: '
    print_freq = args.log_per_step

    # start evaluation
    print_rank_0(" ================ Evaluation pipeline ================ ")
    for iter_i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)  # real is always on the entire dataset

        if num_synthetic < fid_samples:
            model_without_ddp = model.module if isinstance(model, DistributedDataParallel) else model  

            with torch.random.fork_rng(devices=[device]):
                # per node and per step seed
                torch.manual_seed(rng.fold_in(args.seed, rng.get_rank(), iter_i, epoch))

                # prepare
                if args.class_cond:
                    class_labels = F.one_hot(targets, num_classes=args.num_classes,).float()
                else:
                    class_labels = None

                # sample
                with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
                    synthetic_samples = model_without_ddp.sample(
                        samples_shape = samples.shape,
                        model = model_ema,
                        device = device,
                        class_labels = class_labels,
                        )
            torch.cuda.synchronize()

            # ccaling to [0, 1] from [-1, 1]
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

            # upsate FID metric
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)

            # update num_synthetic
            num_synthetic += synthetic_samples.shape[0]

            # save the snapshot if required
            if not snapshots_saved and args.save_image_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.save_image_dir)
                    / "snapshots"
                    / f"{epoch}_{iter_i}{suffix}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.save_image_dir:
                # reshape [bs, c, h, w] -> [bs, h, w, c]
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )

                # save per synthetic image
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.save_image_dir) / f"fid_samples{suffix}"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_utils.get_rank()}_{iter_i}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        PRINT_FREQUENCY = 10
        if (iter_i + 1) % PRINT_FREQUENCY == 0 or iter_i == len(data_loader) - 1:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            print_rank_0(f"Evaluating: current batch {samples.shape[0]},  [{num_synthetic}/{fid_samples}], running fid {running_fid}")

        # debug mode
        if hasattr(args, "debug") and args.debug and iter_i > 50:
            break

    metrics = {"fid": float(fid_metric.compute().detach().cpu())}
    return metrics
