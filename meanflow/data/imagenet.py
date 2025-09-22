import os
import numpy as np
import json
from typing import List

import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class ImageNet1KDataset(data.Dataset):
    def __init__(self,
                 args,
                 img_size: int = 256,
                 is_for_fid = False,
                 transform = None,
                 ):
        super().__init__()
        # ----------------- basic parameters -----------------
        self.is_for_fid = is_for_fid
        self.img_size = img_size

        # load imagenet class names
        abso_path = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(abso_path, 'imagenet_1k_classes.json')
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.classes = data

        # ----------------- dataset & transforms -----------------
        self.image_set = 'train' if is_for_fid else 'train'
        self.data_path = os.path.join(args.root, self.image_set)
        self.transform = transform if transform is not None else self.build_transform()
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, cls_idx = self.dataset[index]
        return image, cls_idx

    def pull_image(self, index):
        # laod data
        image, cls_idx = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = image * 255
        image = np.clip(image, 0., 255.).astype(np.uint8)
        image = image.copy()

        return image, cls_idx

    def build_transform(self,):
        if not self.is_for_fid:
            transforms = T.Compose([
                T.RandomResizedCrop(self.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])
        else:
            t = []
            if self.img_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(self.img_size / crop_pct)
            t.append(
                T.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(T.CenterCrop(self.img_size))
            t.append(T.ToTensor())
            transforms = T.Compose(t)

        return transforms


if __name__ == "__main__":
    import cv2
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='ImageNet-Dataset')

    # opt
    parser.add_argument('--root', default='F:/dataset/imagenet_1k/',
                        help='data root')
    parser.add_argument('--img_size', default=256, type=int,
                        help='input image size.')
    parser.add_argument('--is_for_fid', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    args = parser.parse_args()
  
    # Dataset
    dataset = ImageNet1KDataset(
        args = args,
        img_size = 256,
        is_for_fid = args.is_for_fid,
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

        cv2.imshow(" imagenet image ", image)
        cv2.waitKey(0)
