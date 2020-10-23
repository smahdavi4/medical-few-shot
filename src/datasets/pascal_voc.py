import os

import numpy as np
import torch
import torchvision.transforms.functional as tr_F
from torchvision.datasets.voc import VOCSegmentation
from PIL import Image

from config import cfg
from .base import FewShotDataset

PASCAL_BACKGROUND = 0
PASCAL_VOID = 255

PASCAL_LABELS = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
]

PASCAL_LABEL_NAMES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    'border'
]


class PascalDataset(VOCSegmentation):
    def __init__(self, label_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert 1 <= label_idx <= 20

        # Filter based on id
        self.label_idx = label_idx
        label_name = PASCAL_LABEL_NAMES[label_idx]
        image_set_main = os.path.join(self.root, 'VOCdevkit/VOC2012/ImageSets/Main')

        with open(os.path.join(image_set_main, label_name + '_trainval.txt')) as f:
            file_names = set([x.split()[0].strip() for x in f.readlines() if x.split()[1] == '1'])

        images, masks = [], []
        for image, mask in zip(self.images, self.masks):
            if image[-15:-4] in file_names:
                images.append(image)
                masks.append(mask)

        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        img_np = np.array(img)  # w * h * 3
        img_np = tr_F.to_tensor(img_np)
        img_np = tr_F.normalize(img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target_np = np.array(target)
        fg_target, bg_target = (target_np == self.label_idx).astype(np.bool), np.isin(target_np, [self.label_idx],
                                                                                      invert=True)
        fg_target = torch.tensor(fg_target).float()
        bg_target = torch.tensor(bg_target).float()

        return img_np, fg_target, bg_target


class PascalMultiClassDataset(VOCSegmentation):
    def __init__(self, labels_idx: list, root, image_set='train', transforms=None):
        super().__init__(root, transforms=transforms, image_set=image_set)

        # Filter based on id
        self.labels_idx = labels_idx

        image_set_main = os.path.join(self.root, 'VOCdevkit/VOC2012/ImageSets/Main')

        file_names = set()
        for label_idx in labels_idx:
            label_name = PASCAL_LABEL_NAMES[label_idx]
            with open(os.path.join(image_set_main, label_name + '_{}.txt'.format(image_set))) as f:
                file_names |= set([x.split()[0].strip() for x in f.readlines() if x.split()[1] == '1'])

        images, masks = [], []
        for image, mask in zip(self.images, self.masks):
            if image[-15:-4] in file_names:
                images.append(image)
                masks.append(mask)

        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        img_np = np.array(img)  # w * h * 3
        img_np = tr_F.to_tensor(img_np)
        img_np = tr_F.normalize(img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target_np = np.array(target)
        # Set all other classes to background!
        target_np[np.isin(target_np, [PASCAL_VOID] + self.labels_idx, invert=True)] = PASCAL_BACKGROUND
        target_np = torch.tensor(target_np).long()

        return img_np, target_np


def get_pascal_few_shot_datasets(labels, iterations, N_shot, N_query, transforms):
    pascals = [PascalDataset(label, cfg['voc']['root'], transforms=transforms) for label in labels]
    dataset = FewShotDataset(pascals, iterations, N_shot, N_query)
    return dataset
