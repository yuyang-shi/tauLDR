import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import dataset_utils
import numpy as np
import torchvision.datasets
import torchvision.transforms
import os
import random
from PIL import Image


@dataset_utils.register_dataset
class DiscreteMNIST(torchvision.datasets.MNIST):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
                         download=cfg.data.download)

        self.data = self.data.unsqueeze(1).to(device)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        return img


@dataset_utils.register_dataset
class ConditionalDiscreteMNIST(torchvision.datasets.MNIST):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
                         download=cfg.data.download)

        self.data = self.data.unsqueeze(1).to(device)

        self.mask = torch.zeros_like(self.data[0])
        self.mask[:, (self.data.shape[-2]//4):(-self.data.shape[-2]//4), (self.data.shape[-1]//4):(-self.data.shape[-1]//4)] = 1

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        cond = img * (1 - self.mask)

        img = torch.cat([cond, img], dim=0)
        return img


@dataset_utils.register_dataset
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, train=cfg.data.train,
            download=cfg.data.download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.to(device).view(-1, 3, 32, 32)

        self.random_flips = cfg.data.random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()


    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img


@dataset_utils.register_dataset
class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S
        L = cfg.data.shape[0]
        np_data = np.load(cfg.data.path) # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]


@dataset_utils.register_dataset
class ConditionalDiscreteImageNet(torchvision.datasets.ImageNet):
    def __init__(self, cfg, device):
        super().__init__(root=cfg.data.root, split='train' if cfg.data.train else 'val')
        self.train = cfg.data.train
        self.high_resolution = cfg.data.high_resolution
        self.low_resolution = cfg.data.low_resolution
        self.random_flips = cfg.data.random_flips
        self.device = device
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        arr = center_crop_arr(sample, self.high_resolution)
        arr = np.transpose(arr, [2, 0, 1])  # high res
        high_res = torch.from_numpy(arr)

        if self.random_flips and random.random() < 0.5 and self.train:
            high_res = high_res.flip(-1)
        
        low_res = F.interpolate(F.interpolate(high_res.float().unsqueeze(0), self.low_resolution, mode="area"), 
                                self.high_resolution, mode="area").clip(0, 255).byte().squeeze(0)

        return torch.cat([low_res, high_res], 0).to(self.device), torch.tensor(target).to(self.device)


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
