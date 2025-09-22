import os
import numpy as np
from typing import List

import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


class CifarDataset(data.Dataset):
    def __init__(self,
                 img_size: int = 32,
                 is_for_fid = False,
                 transform = None,
                 ):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.is_for_fid = is_for_fid
        self.cifar10_classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]

        # ----------------- dataset & transforms -----------------
        self.image_set = 'train' if is_for_fid else 'train'
        self.transform = transform if transform is not None else self.build_transform()
        abso_path = os.path.dirname(os.path.abspath(__file__))
        if is_for_fid:
            self.dataset = CIFAR10(os.path.join(abso_path, 'cifar_data/'), train=True, download=True, transform=self.transform)
        else:
            self.dataset = CIFAR10(os.path.join(abso_path, 'cifar_data/'), train=True, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, cls_idx = self.dataset[index]

        return image, cls_idx
    
    def pull_image(self, index):
        # laod data
        image, cls_idx = self.dataset[index]

        # convert image tensor into image numpy
        image = image.permute(1, 2, 0).numpy()
        image = image * 255
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        return image, cls_idx

    def build_transform(self):
        if self.is_for_fid:
            transforms = T.Compose([T.ToTensor(),])
        else:
            transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(),])

        return transforms

if __name__ == "__main__":
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser(description='Cifar10 Dataset')

    # opt
    parser.add_argument('--img_size', default=32, type=int,
                        help='data root')
    parser.add_argument('--is_for_fid', action="store_true", default=False,
                        help='mixup augmentation.')

    args = parser.parse_args()

    # dataset
    dataset = CifarDataset(
        img_size = 32,
        is_for_fid = args.is_for_fid
        )  
    print('Dataset size: ', len(dataset))

    for i in range(len(dataset)):
        image, cls_idx = dataset[i]

        # convert image tensor into image numpy
        image = image.permute(1, 2, 0).numpy()
        image = image * 255.
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        # to BGR
        image = image[..., (2, 1, 0)]

        cv2.imshow(" cifar image ", image)
        cv2.waitKey(0)
