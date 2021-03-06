import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize

from config import cfg
from datasets.visceral import get_visceral_medical_few_shot_dataset
from models.panet import FewShotSeg as PANetFewShotSeg
from utils.colab import load_state, save_state
from utils.metric import Metric


def train_medical(device, resume=False):
    assert cfg['panet']['backbone'] == 'unet'

    dataset_name = 'ircadb'
    model = PANetFewShotSeg(in_channels=cfg[dataset_name]['channels'], pretrained_path=None, cfg={'align': True},
                            encoder_type=cfg['panet']['backbone']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['panet']['lr'], momentum=cfg['panet']['momentum'],
                                weight_decay=cfg['panet']['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=cfg['panet']['lr_milestones'], gamma=0.1)
    epoch = 0
    model.train()

    if resume:
        epoch = load_state(cfg[dataset_name]['model_name'], model, optimizer, scheduler)

    transforms = Compose(
        [
            Resize(size=cfg['panet']['unet_inp_size']),
        ]
    )

    train_dataset = get_visceral_medical_few_shot_dataset(
        organs=["lungs", "kidneys", "spleen", "psoas_majors"],
        patient_ids=range(0, 16),
        shots=5,
        mode='train',
        transforms=transforms
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss()

    log_loss = {'loss': 0, 'align_loss': 0}
    i_iter = 0
    while True:
        for i_iter_, (support, query) in enumerate(tqdm(trainloader, position=0, leave=True)):
            i_iter += 1
            support_images = [[]]
            support_fg_mask = [[]]
            support_bg_mask = [[]]
            # for i in range(len(support)):
            support_bg = support[1] == 0
            support_fg = support[1] == 1
            support_images[0].append(support[0].to(device))
            support_fg_mask[0].append(support_fg.float().to(device))
            support_bg_mask[0].append(support_bg.float().to(device))

            query_images = []
            query_labels = []

            # for i in range(len(query)):
            query_images.append(query[0].to(device))
            query_labels.append(query[1].to(device))

            query_labels = torch.cat(query_labels, dim=0).long().to(device)

            # Forward and Backward
            optimizer.zero_grad()
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                           query_images)
            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss * cfg['panet']['align_loss_scalar']
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % cfg['panet']['save_period'] == 0:
                loss = log_loss['loss'] / (i_iter + 1)
                align_loss = log_loss['align_loss'] / (i_iter + 1)
                print('\nstep {}: loss: {}, align_loss: {}'.format(i_iter + 1, loss, align_loss))
            if (i_iter + 1) % cfg['panet']['save_period'] == 0:
                save_state(cfg[dataset_name]['model_name'], model, optimizer, scheduler, epoch + i_iter + 1)
                print("\nModel Saved On Iteration {} ...".format(epoch + i_iter + 1))

    return model


def test_medical(device, model=None, test_organ='liver'):
    assert cfg['panet']['backbone'] == 'unet'

    dataset_name = 'ircadb'

    if model is None:
        model = PANetFewShotSeg(in_channels=cfg[dataset_name]['channels'], pretrained_path=None, cfg={'align': True},
                                encoder_type=cfg['panet']['backbone']).to(device)
        load_state(cfg[dataset_name]['model_name'], model)

    model.eval()

    transforms = Compose(
        [
            Resize(size=cfg['panet']['unet_inp_size']),
        ]
    )

    test_dataset = get_visceral_medical_few_shot_dataset(
        organs=[test_organ],
        patient_ids=range(16, 20),
        shots=5,
        mode='test',
        transforms=transforms
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    metric = Metric(max_label=20, n_runs=1)
    for i_iter, (support, query) in enumerate(tqdm(testloader, position=0, leave=True)):
        support_images = [[]]
        support_fg_mask = [[]]
        support_bg_mask = [[]]
        # for i in range(len(support)):
        support_bg = support[1] == 0
        support_fg = support[1] == 1
        support_images[0].append(support[0].to(device))
        support_fg_mask[0].append(support_fg.float().to(device))
        support_bg_mask[0].append(support_bg.float().to(device))

        query_images = []
        query_labels = []

        # for i in range(len(query)):
        query_images.append(query[0].to(device))
        query_labels.append(query[1].to(device))

        query_labels = torch.cat(query_labels, dim=0).long().to(device)

        query_pred, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)

        print("Support ", i_iter)
        plt.subplot(1, 2, 1)
        try:
            plt.imshow(np.moveaxis(support[0].squeeze().cpu().detach().numpy(), 0, 2))
        except np.AxisError:
            plt.imshow(support[0].squeeze().cpu().detach().numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(support[1].squeeze())
        plt.show()

        print("Query ", i_iter)

        plt.subplot(1, 3, 1)
        try:
            plt.imshow(np.moveaxis(query[0].squeeze().cpu().detach().numpy(), 0, 2))
        except np.AxisError:
            plt.imshow(query[0].squeeze().cpu().detach().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(query[1].squeeze())
        plt.subplot(1, 3, 3)
        plt.imshow(np.array(query_pred.argmax(dim=1)[0].cpu()))
        metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                      np.array(query_labels[0].cpu()), n_run=0)
        plt.show()

    classIoU, meanIoU = metric.get_mIoU(n_run=0)
    classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=0)

    print('classIoU', classIoU.tolist())
    print('meanIoU', meanIoU.tolist())
    print('classIoU_binary', classIoU_binary.tolist())
    print('meanIoU_binary', meanIoU_binary.tolist())
    print('classIoU: {}'.format(classIoU))
    print('meanIoU: {}'.format(meanIoU))
    print('classIoU_binary: {}'.format(classIoU_binary))
    print('meanIoU_binary: {}'.format(meanIoU_binary))
