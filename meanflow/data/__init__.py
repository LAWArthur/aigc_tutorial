import torch.utils.data as data

from .cifar import CifarDataset
from .imagenet import ImageNet1KDataset


def build_dataset(args, transform=None, is_for_fid=False):
    if args.dataset == 'cifar10':
        args.img_dim = 3
        args.num_classes = 10
        return CifarDataset(
            img_size = 32,
            is_for_fid = is_for_fid,
            transform = transform,
            )
    
    elif args.dataset == 'imagenet1k':
        args.img_dim = 3
        args.num_classes = 1000
        return ImageNet1KDataset(
            img_size = args.img_size,
            is_for_fid = is_for_fid,
            transform = transform,
            )
        
    raise NotImplementedError(f" > Unknown dataset: {args.dataset}")
    

def build_dataloader(args, dataset, is_for_fid=False):
    sampler = data.distributed.DistributedSampler(dataset) if args.distributed else data.RandomSampler(dataset)
    per_gpu_batch_size = args.batch_size // args.world_size
    batch_sampler = data.BatchSampler(
        sampler = sampler,
        batch_size = per_gpu_batch_size,
        drop_last = False if not is_for_fid else True,
        )
    dataloader = data.DataLoader(
        dataset,
        batch_sampler = batch_sampler,
        num_workers = args.num_workers,
        pin_memory = False,
        )

    return dataloader
