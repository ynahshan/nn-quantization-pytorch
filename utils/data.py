import os
import torchvision.datasets as datasets

__DATASETS_DEFAULT_PATH = '/tmp/Datasets/'

from pathlib import Path
home = str(Path.home())
__IMAGENET_DEFAULT_PATH = '/home/cvds_lab/datasets/ILSVRC2012/'

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True,
                datasets_path=None):
    train = (split == 'train')
    root = os.path.join(datasets_path if datasets_path is not None else __DATASETS_DEFAULT_PATH, name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if datasets_path is None:
            datasets_path = __IMAGENET_DEFAULT_PATH
        if train:
            root = os.path.join(datasets_path, 'train')
        else:
            root = os.path.join(datasets_path, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
