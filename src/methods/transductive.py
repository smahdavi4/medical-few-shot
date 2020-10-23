import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from config import cfg
from datasets.visceral import get_visceral_medical_few_shot_dataset
from lightning_trainers.transductive import MetaLearnTransductive, META_TEST_ORGANS, META_TEST_PATIENTS, \
    MetaTestTransductive, META_TEST_CLASSES
from utils.dice import dice_score_perclass
from utils.image import CropOrPad


def train_transductive(device, resume=False):
    train_transforms = Compose(
        [
            CropOrPad(cfg['transductive']['unet_inp_size']),
            # RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip(p=0.5),
        ]
    )
    test_transforms = Compose(
        [
            CropOrPad(cfg['transductive']['unet_inp_size']),
        ]
    )
    checkpoint_path = os.path.join(cfg['base_models_path'], cfg['visceral']['model_pretrained_name'])
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='min_dice', mode='max')
    if resume:
        resume_from_checkpoint = checkpoint_path
    else:
        resume_from_checkpoint = None
    model = MetaLearnTransductive(train_transforms=train_transforms, test_transforms=test_transforms)
    trainer = Trainer(gpus=1, progress_bar_refresh_rate=10, max_epochs=cfg['transductive']['pretrain_epochs'],
                      checkpoint_callback=checkpoint_callback, resume_from_checkpoint=resume_from_checkpoint)
    trainer.fit(model)
    # trainer.test()


def test_transductive(device, train_checkpoint_path, epochs=5):
    transforms = Compose(
        [
            CropOrPad(cfg['transductive']['unet_inp_size']),
        ]
    )

    dataset = get_visceral_medical_few_shot_dataset(
        organs=META_TEST_ORGANS,
        patient_ids=META_TEST_PATIENTS,
        shots=10,
        transforms=transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    for i_iter, (support, query) in enumerate(tqdm(loader, position=0, leave=True)):
        # Fetch images and masks
        support_images = []
        support_masks = []

        query_images = []
        query_masks = []

        for i in range(len(support)):
            support_images.append(support[i][0].to(device))
            support_masks.append(support[i][1].to(device))

        for i in range(len(query)):
            query_images.append(query[i][0].to(device))
            query_masks.append(query[i][1].to(device))

        support_images = torch.cat(support_images, dim=0).to(device)
        query_images = torch.cat(query_images, dim=0).to(device)

        query_masks = torch.cat(query_masks, dim=0).long().to(device)
        support_masks = torch.cat(support_masks, dim=0).long().to(device)

        # Create model
        checkpoint_backbone = MetaLearnTransductive.load_from_checkpoint(train_checkpoint_path,
                                                                         train_transforms=transforms,
                                                                         test_transforms=transforms)
        model = MetaTestTransductive(checkpoint_backbone, support_images, support_masks)
        trainer = Trainer(gpus=1, progress_bar_refresh_rate=1, max_epochs=epochs, checkpoint_callback=None)
        trainer.fit(model)

        for i in range(0, len(query_images), 10):
            query_chunk_images = query_images[i:i + 10]
            query_pred = model(query_chunk_images)

            print("Query chunk {} Dice:".format(i / 10))
            query_dice = dice_score_perclass(torch.argmax(query_pred, dim=1), query_chunk_images,
                                             len(META_TEST_CLASSES))
            for cls_name, cls_dice in zip(META_TEST_CLASSES, query_dice):
                print("QUERY: class: <{}> dice: <{}>".format(cls_name, cls_dice))
