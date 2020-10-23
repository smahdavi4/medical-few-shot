import numpy as np

import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch

from datasets.visceral import get_normal_medical_dataset, CLASS_WEIGHTS
from models.unet import UNetMedicalMedium

from utils.dice import dice_score_perclass

META_TRAIN_BATCH_SIZE = 5
META_TRAIN_ORGANS = ["kidneys", "spleen", "psoas_majors"]
META_TRAIN_CLASSES = ["background"] + META_TRAIN_ORGANS
META_TRAIN_CLASSES_WEIGHTS = np.array([CLASS_WEIGHTS[cls] for cls in META_TRAIN_CLASSES])
META_TRAIN_CLASSES_WEIGHTS_NORMALIZED = np.max(META_TRAIN_CLASSES_WEIGHTS) / META_TRAIN_CLASSES_WEIGHTS
META_TRAIN_PATIENS = {
    'train': range(0, 15),
    'val': range(15, 17),
    'test': range(17, 20)
}

META_TEST_ORGANS = ["liver"]
META_TEST_PATIENTS = range(17, 20)
META_TEST_CLASSES = ["background"] + META_TEST_ORGANS
META_TEST_CLASSES_WEIGHTS = np.array([CLASS_WEIGHTS[cls] for cls in META_TEST_CLASSES])
META_TEST_CLASSES_WEIGHTS_NORMALIZED = np.max(META_TEST_CLASSES_WEIGHTS) / META_TEST_CLASSES_WEIGHTS


class MetaLearnTransductive(pl.LightningModule):
    def __init__(self, train_transforms, test_transforms):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        # self.unet = UnetPlusPlus(num_classes=len(META_TRAIN_ORGANS) + 1, input_channels=1)
        # self.unet = UNetBackbone(in_channels=1, out_channels=len(META_TRAIN_CLASSES))
        self.unet = UNetMedicalMedium(in_ch=1, out_ch=len(META_TRAIN_CLASSES))
        self.class_weights = torch.tensor(META_TRAIN_CLASSES_WEIGHTS_NORMALIZED).float().cuda()

    def forward(self, input):
        return self.unet(input)

    def training_step(self, batch, batch_idx):
        X, y = batch
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        y_pred = self(X)
        loss = criterion(y_pred, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs,
                'dice': dice_score_perclass(torch.argmax(y_pred, dim=1), y, len(META_TRAIN_CLASSES))}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['dice'] for x in outputs]).mean(dim=0)
        tensorboard_logs = {'train_loss': avg_loss}
        print("")
        for i in range(len(META_TRAIN_CLASSES)):
            tensorboard_logs['train_{}'.format(META_TRAIN_CLASSES[i])] = avg_dice[i]
            print("TRAIN: Dice for organ <{}> : {}".format(META_TRAIN_CLASSES[i], avg_dice[i]))
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        X, y = batch
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        y_pred = self(X)
        val_loss = criterion(y_pred, y)
        return {'val_loss': val_loss,
                'dice': dice_score_perclass(torch.argmax(y_pred, dim=1), y, len(META_TRAIN_CLASSES))}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['dice'] for x in outputs]).mean(dim=0)
        tensorboard_logs = {'val_loss': avg_loss}
        print("")
        min_dice = 100
        for i in range(len(META_TRAIN_CLASSES)):
            tensorboard_logs['val_{}'.format(META_TRAIN_CLASSES[i])] = avg_dice[i]
            print("VAL Dice for organ <{}> : {}".format(META_TRAIN_CLASSES[i], avg_dice[i]))
            min_dice = min(min_dice, avg_dice[i])
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'min_dice': min_dice}

    def test_step(self, batch, batch_idx):
        X, y = batch
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        y_pred = self(X)
        test_loss = criterion(y_pred, y)
        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        return [optimizer], [scheduler]

    def prepare_data(self):
        self.visceral_train = get_normal_medical_dataset(META_TRAIN_ORGANS, META_TRAIN_PATIENS['train'],
                                                         self.train_transforms)
        self.visceral_val = get_normal_medical_dataset(META_TRAIN_ORGANS, META_TRAIN_PATIENS['val'],
                                                       self.test_transforms)
        self.visceral_test = get_normal_medical_dataset(META_TRAIN_ORGANS, META_TRAIN_PATIENS['test'],
                                                        self.test_transforms)

    def train_dataloader(self):
        loader = DataLoader(self.visceral_train, batch_size=META_TRAIN_BATCH_SIZE, num_workers=0, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.visceral_val, batch_size=META_TRAIN_BATCH_SIZE, num_workers=0)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.visceral_test, batch_size=META_TRAIN_BATCH_SIZE, num_workers=0)
        return loader


class MetaTestTransductive(pl.LightningModule):
    def __init__(self, unet_backbone, support_images, support_masks):
        super().__init__()
        self.unet = unet_backbone
        self.class_weights = torch.tensor(META_TEST_CLASSES_WEIGHTS_NORMALIZED).float().cuda()
        self.Conv3x3 = nn.Conv2d(len(META_TRAIN_CLASSES), len(META_TEST_CLASSES), kernel_size=3, stride=1, padding=1)

        self.support_images = support_images
        self.support_masks = support_masks

    def forward(self, input):
        x = self.unet(input)
        x = self.Conv3x3(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        y_pred = self(X)
        loss = criterion(y_pred, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs,
                'dice': dice_score_perclass(torch.argmax(y_pred, dim=1), y, len(META_TEST_CLASSES))}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['dice'] for x in outputs]).mean(dim=0)
        tensorboard_logs = {'train_loss': avg_loss}
        print("")
        for i in range(len(META_TEST_CLASSES)):
            tensorboard_logs['train_{}'.format(META_TEST_CLASSES[i])] = avg_dice[i]
            print("TRAIN: Dice for organ <{}> : {}".format(META_TEST_CLASSES[i], avg_dice[i]))
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        return [optimizer], [scheduler]

    def prepare_data(self):
        self.dataset = TensorDataset(self.support_images, self.support_masks)

    def train_dataloader(self):
        loader = DataLoader(self.dataset, shuffle=True, batch_size=5)
        return loader
