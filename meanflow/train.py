import os
import time
import copy
import argparse
import datetime
from functools import partial
import rng

# ---------------- Torch compoments ----------------
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------- Dataset compoments ----------------
from data import build_dataset

# ---------------- Model compoments ----------------
from meanflow_pipe import instantiate_meanflow

# ---------------- Utils compoments ----------------
from utils import distributed_utils
from utils.misc import setup_seed, print_rank_0, save_model, load_model

# ---------------- Training engine ----------------
from engine import train_step, train_one_epoch, eval_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic settings
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument("--compile", action="store_true", default=False, help="Unable compilation.")
    parser.add_argument('--debug', action='store_true', default=False,
                        help='distributed training')
    
    # Epoch settings
    parser.add_argument('--epochs', type=int, default=16000,
                        help='number of total epochs')
    parser.add_argument('--warmup_epochs', type=int, default=200,
                        help='number of warmup epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    
    # Input settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset name')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='path to dataset folder')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='gradient accumulation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--img_size', default=224, type=int, 
                        help='input image size.')
    parser.add_argument('--img_dim', default=3, type=int, 
                        help='input image size.')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help='path to save trained model.')
    parser.add_argument('--save_image_dir', type=str, default='results/',
                        help='path to save trained model.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_ckpt_num', default=5, type=int, 
                        help='number of the saved checkpoint.')
    parser.add_argument('--save_ckpt_freq', default=20, type=int, 
                        help='frequency of the saved checkpoint.')
    parser.add_argument("--log_per_step", default=100, type=int, metavar="N",
                        help="Log training stats every N iterations",)

    # Model settings
    parser.add_argument('--model', type=str, default='dhw_unet',
                        help='model name')
    parser.add_argument('--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Log training stats every N iterations",)
    parser.add_argument('--class_cond', action='store_true', default=False,
                        help='enable class condition')

    # MeanFlow specific settings
    parser.add_argument("--ratio", default=0.75, type=float,
                        help="Probability of sampling r (or h) DIFFERENT from t")  
    parser.add_argument("--tr_sampler", default="v1", type=str, choices=["v0", "v1"],
                        help="Joint (t, r) sampler version.")
    
    parser.add_argument("--P_mean_t", default=-0.6, type=float,
                        help="P_mean_t of lognormal sampler.")
    parser.add_argument("--P_std_t", default=1.6, type=float,
                        help="P_std_t of lognormal sampler.")
    parser.add_argument("--P_mean_r", default=-4.0, type=float,
                        help="P_mean_r of lognormal sampler.")
    parser.add_argument("--P_std_r", default=1.6, type=float,
                        help="P_std_r of lognormal sampler.")
    
    parser.add_argument("--norm_p", default=0.75, type=float,
                        help="Norm power for adaptive weight.")
    parser.add_argument("--norm_eps", default=1e-3, type=float,
                        help="Small constant for adaptive weight division.")
    parser.add_argument("--use_edm_aug", action="store_true", default=False,
                        help="Enable EDM augmentation with augment labels as conditions.")

    # Optimizer settings
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="Exponential moving average decay rate.")
    parser.add_argument("--optimizer_betas", default=[0.9, 0.999], nargs="+", type=float,
                        help="beta1 and beta2 for Adam optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay for Adam optimizer")

    # ModelEMA settings
    parser.add_argument("--ema_decay", default=0.9999, type=float,
                        help="Exponential moving average decay rate.")
    parser.add_argument("--ema_decays", default=[0.99995, 0.9996], nargs="+", type=float,
                        help="Extra EMA decay rates.")

    # Evaluation settings
    parser.add_argument("--fid_samples", default=50000, type=int,
                        help="number of synthetic samples for FID evaluations")
    parser.add_argument("--eval_only", action="store_true",
                        help="No training, only run evaluation")
    parser.add_argument("--compute_fid", action="store_true",
                        help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.")
    parser.add_argument("--save_fid_samples", action="store_true", help="Save all samples generated for FID computation.")

    # DDP
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='the number of local rank.')

    return parser.parse_args()


def get_data_loader(args, dataset, is_for_fid=False):
    num_tasks = distributed_utils.get_world_size()
    global_rank = distributed_utils.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        sampler = sampler,
        worker_init_fn = partial(rng.worker_init_fn, rank=global_rank),
        batch_size = args.batch_size // args.world_size,
        num_workers = args.num_workers,
        pin_memory = False,
        drop_last = not is_for_fid,  # for FID evaluation, we want to keep all samples
    )

    return data_loader


def main():
    args = parse_args()
    # set random seed
    setup_seed(args.seed)

    # Path to save model
    output_dir = os.path.join(args.output_dir, args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # ------------------------- Build DDP environment -------------------------
    ## LOCAL_RANK is the global GPU number tag, the value range is [0, world_size - 1].
    ## LOCAL_PROCESS_RANK is the number of the GPU of each machine, not global.
    local_rank = local_process_rank = -1
    print_rank_0(" ================= DDP settings ================= ", local_rank)
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
        try:
            # Multiple Mechine & Multiple GPUs (world size > 8)
            local_rank = torch.distributed.get_rank()
            local_process_rank = int(os.getenv('LOCAL_PROCESS_RANK', '0'))
        except:
            # Single Mechine & Multiple GPUs (world size <= 8)
            local_rank = local_process_rank = torch.distributed.get_rank()
    print_rank_0(args, local_rank)
    args.world_size = distributed_utils.get_world_size()
    print_rank_0(' > World size: {}'.format(distributed_utils.get_world_size()), local_rank)
    print_rank_0(" > LOCAL RANK: ", local_rank)
    print_rank_0(" > LOCAL_PROCESS_RANL: ", local_process_rank)

    # ------------------------- Setup CUDA -------------------------
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        print_rank_0(' > There is no available GPU.')
        device = torch.device("cpu")

    # ------------------------- Setup Tensorboard -------------------------
    tblogger = None
    if local_rank <= 0 and args.tfboard:
        print_rank_0(' > launch tensorboard', local_rank)
        from torch.utils.tensorboard import SummaryWriter
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, args.model, time_stamp)
        print_rank_0(" > Tensorboard writer created at ", log_path)
        os.makedirs(log_path, exist_ok=True)
        tblogger = SummaryWriter(log_path)

    # ------------------------- Build Dataset -------------------------
    train_dataset = build_dataset(args, is_for_fid=False)
    valid_dataset = build_dataset(args, is_for_fid=True)
    train_dataloader = get_data_loader(args, train_dataset, is_for_fid=False)
    valid_dataloader = get_data_loader(args, valid_dataset, is_for_fid=True)

    print_rank_0(' =================== Dataset Information =================== ', local_rank)
    print_rank_0(f" > Training data size: {len(train_dataset)}", local_rank)
    print_rank_0(f" > Validation data size: {len(valid_dataset)}", local_rank)

    # ------------------------- Setup Model -------------------------
    print_rank_0(f" > create model: {args.model}", local_rank)
    model = instantiate_meanflow(args)
    model.to(device)
    model_without_ddp = model

    # ------------------------- Setup Optimzier -------------------------
    optimizer = torch.optim.AdamW(
        params = model.parameters(),
        lr = args.lr,
        betas = args.optimizer_betas,
        weight_decay = args.weight_decay,
        )

    # ------------------------- Setup LR-Scheduler -------------------------
    warmup_iters = args.warmup_epochs * len(train_dataloader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters,)
    # main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs * len(train_dataloader), factor=1.0)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader), eta_min=args.lr * 0.25)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])
    
    # keep training from the given checkpoint
    print_rank_0(" > Keep training from the checkpoint: {}".format(args.resume), local_rank)
    load_model(
        args = args,
        model_without_ddp = model_without_ddp,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
    )

    # ------------------------- Setup DDP Model -------------------------
    if args.distributed:
        model = DDP(
            model, device_ids=[args.gpu],
            find_unused_parameters = False,
            broadcast_buffers = False,
            static_graph = True,
            gradient_as_bucket_view = True,
            )
        model_without_ddp = model.module

    # ------------------------- Training Pipeline -------------------------
    compiled_train_step = torch.compile(
        train_step,
        disable = not args.compile,
    )

    start_time = time.time()
    print_rank_0(" ========= Start training for {} epochs ========= ".format(args.epochs), local_rank)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        train_one_epoch(
            args = args,
            device = device,
            model = model,
            compiled_train_step = compiled_train_step,
            data_loader = train_dataloader,
            optimizer = optimizer,
            lr_schedule = lr_scheduler,
            epoch = epoch,
            local_rank = local_rank,
            tblogger = tblogger,
            )

        # evaluate
        if (epoch % args.save_ckpt_freq) == 0 or (epoch + 1 == args.epochs):            
            # Save model
            print(' > Saving the model at epoch-{} ...'.format(epoch))
            save_model(
                args = args,
                model_without_ddp = model_without_ddp,
                optimizer = optimizer,
                lr_scheduler = lr_scheduler,
                epoch = epoch,
                local_rank = local_rank,
                )
            
            # Eval no-ema model:
            model_eval = model_without_ddp.model
            eval_stats = eval_one_epoch(
                args = args,
                model = model,
                model_ema = model_eval,
                data_loader = valid_dataloader,
                device = device,
                epoch = epoch,
                suffix='_noema'
                )
            if tblogger is not None and "fid" in eval_stats:
                print(f"Eval {epoch + 1} epochs finished: FID w/o ema: {eval_stats['fid']}")
                tblogger.add_scalar("FID", eval_stats["fid"], epoch + 1)

            # Eval ModelEMA
            model_eval = model_without_ddp.model_ema
            ema_decay = model_eval.ema_decay
            eval_stats = eval_one_epoch(
                args = args,
                model = model,
                model_ema = model_eval,
                data_loader = valid_dataloader,
                device = device,
                epoch = epoch,
                suffix = f'_ema{ema_decay}',
                )
            if "fid" in eval_stats:
                print(f"Eval {epoch + 1} epochs finished: FID_ema{ema_decay}: {eval_stats['fid']}")
            if tblogger is not None:
                tblogger.add_scalar(f"FID_ema{ema_decay}", eval_stats["fid"], epoch + 1)

            # Eval extra ema model:
            for i in range(len(model_without_ddp.ema_decays)):
                model_eval = model_without_ddp._modules[f"model_ema{i + 1}"]
                ema_decay = model_eval.ema_decay
                eval_stats = eval_one_epoch(
                    args = args,
                    model = model,
                    model_ema = model_eval,
                    data_loader = valid_dataloader,
                    device = device,
                    epoch = epoch,
                    suffix = f'_ema{ema_decay}',
                    )
                if "fid" in eval_stats:
                    print(f"Eval {epoch + 1} epochs finished: FID_ema{ema_decay}: {eval_stats['fid']}")
                if tblogger is not None:
                    tblogger.add_scalar(f"FID_ema{ema_decay}", eval_stats["fid"], epoch + 1)

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
