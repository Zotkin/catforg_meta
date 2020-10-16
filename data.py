from torch.utils.data import Dataset
import torch
from typing import Tuple, Dict, List, Sequence
import numpy as np
from PIL import Image


class SupervisedImageDataset(Dataset):

    def __init__(self, img: np.ndarray, label: np.ndarray, transforms) -> None:
        self.img = img
        self.label = label.astype(np.uint8)
        self.transforms = transforms

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.uint8]:
        img = self.img[idx]

        img = Image.fromarray(img)
        img = self.transforms(img)
        label = self.label[idx]
        return img, label


class DoubleAugmentedCIFAR10(Dataset):

    def __init__(self, img: np.ndarray, transforms):
        self.img = img
        self.transforms = transforms

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = self.img[idx]
        img = Image.fromarray(img)
        img_aug_1 = self.transforms(img)
        img_aug_2 = self.transforms(img)

        return (img_aug_1, img_aug_2)