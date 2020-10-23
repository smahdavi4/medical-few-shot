import os
import logging
import gc

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_ct import read_data
from utils.constants import image_shape
from utils.dataset import InterPatientDataset
from utils.losses import ncc_loss, binary_dice_loss, gradient_loss, get_recon_loss
from utils.colab import RESULT_FOLDER, download_results
from utils.evaluate import save_losses, report_few_shot_test_organs, report_support_recon_loss_organs
from model import RegModel
from config import cfg, set_cfg

# Log to both file and stdout
from utils.parsing import str2bool

logging.basicConfig(filename=os.path.join(RESULT_FOLDER, 'logs.log'), level=logging.INFO,
                    format='%(asctime)s: %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(0)
torch.manual_seed(0)


def calculate_loss(fixed_image, fixed_seg, moving_image, moving_seg, moved_image, reg_model, disp):
    """
    Calculates the final loss based on the given config.
    """
    recon_loss = get_recon_loss()
    if cfg['use_smooth_loss']:
        grad_loss = gradient_loss(disp)
        loss_us = recon_loss(moved_image, fixed_image) + cfg['gamma'] * grad_loss
    else:
        grad_loss = torch.zeros(1)  # No loss
        loss_us = recon_loss(moved_image, fixed_image)

    if cfg['use_seg_loss']:
        loss_dices = []
        for cls in cfg['training_classes']:
            fixed_y_true = (fixed_seg == cls).to(device).float()
            moving_y = (moving_seg == cls).to(device).float()
            fixed_y_pred = reg_model.spatial_transform(moving_y, disp)
            loss_dices.append(binary_dice_loss(fixed_y_true, fixed_y_pred))

        loss_seg = sum(loss_dices) / len(cfg['training_classes'])
        loss = loss_us + cfg['lamda'] * loss_seg
    else:
        loss_seg = torch.zeros(1)  # No loss
        loss = loss_us
    return loss, loss_us, loss_seg, grad_loss


def train_network():
    (X_train, y_train), (X_val, y_val), _ = read_data(
        visceral_path="./visceral_silver_1.5mm.h5",
        image_shape=image_shape,
        split=[58, 5, 0]
    )

    ndims = 2
    nb_enc_features = [32, 32, 64, 128]
    nb_dec_features = [128, 128, 64, 32, 32, 16]

    logging.info("UNet features: {}, {}".format(nb_enc_features, nb_dec_features))

    # inputs: moving_image, fixed_image
    reg_model = RegModel(ndims, image_shape, nb_enc_features, nb_dec_features).to(device).float()

    # test
    tgt, src = torch.rand(1, 1, 256, 256).to(device).float(), torch.rand(1, 1, 256, 256).to(device).float()
    moved, disp = reg_model(src, tgt)

    assert tuple(moved.shape) == (1, 1, 256, 256)
    assert tuple(disp.shape) == (1, 2, 256, 256)

    optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-3)

    # dataset = Dataset(X_train, diff_min, diff_max, steps_per_epoch, batch_size)
    dataset = InterPatientDataset(X_train, y_train, cfg['diff_min'], cfg['diff_max'], cfg['steps_per_epoch'],
                                  cfg['batch_size'])
    dataset_val = InterPatientDataset(X_val, y_val, cfg['diff_min'], cfg['diff_max'], cfg['steps_per_epoch'],
                                      cfg['batch_size'])

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=6)
    dataloader_val = DataLoader(dataset_val, batch_size=cfg['batch_size'], num_workers=6)

    losses = []
    val_losses = []
    for epoch in range(cfg['nb_epochs']):
        running_loss = 0
        running_val_loss = 0
        running_seg_loss = 0
        running_seg_val_loss = 0
        running_grad_loss = 0
        running_grad_val_loss = 0

        # Train
        reg_model.train()
        for i, data in enumerate(dataloader):
            moving_image, moving_seg, fixed_image, fixed_seg = data

            moving_image = moving_image.to(device).float()
            moving_seg = moving_seg.to(device).long()
            fixed_image = fixed_image.to(device).float()
            fixed_seg = fixed_seg.to(device).long()

            optimizer.zero_grad()
            moved_image, disp = reg_model(moving_image, fixed_image)
            loss, loss_us, loss_seg, grad_loss = calculate_loss(fixed_image, fixed_seg, moving_image, moving_seg,
                                                                moved_image, reg_model, disp)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_seg_loss += loss_seg.item()
            running_grad_loss += grad_loss.item()

        # Val
        reg_model.eval()
        for i, data in enumerate(dataloader_val):
            moving_image, moving_seg, fixed_image, fixed_seg = data

            moving_image = moving_image.to(device).float()
            moving_seg = moving_seg.to(device).long()
            fixed_image = fixed_image.to(device).float()
            fixed_seg = fixed_seg.to(device).long()

            moved_image, disp = reg_model(moving_image, fixed_image)
            loss, loss_us, loss_seg, grad_loss = calculate_loss(fixed_image, fixed_seg, moving_image, moving_seg,
                                                                moved_image, reg_model, disp)

            running_val_loss += loss.item()
            running_seg_val_loss += loss_seg.item()
            running_grad_val_loss += grad_loss.item()

        epoch_loss = running_loss / cfg['steps_per_epoch']
        epoch_seg_loss = running_seg_loss / cfg['steps_per_epoch']
        epoch_grad_loss = running_grad_loss / cfg['steps_per_epoch']

        epoch_val_loss = running_val_loss / cfg['steps_per_epoch']
        epoch_seg_val_loss = running_seg_val_loss / cfg['steps_per_epoch']
        epoch_grad_val_loss = running_grad_val_loss / cfg['steps_per_epoch']

        losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        logging.info(
            'Epoch %d: loss=%.10f , seg_loss:%.10f , grad_loss:%.10f | val loss=%.10f val_seg_loss:%.10f val_grad_loss:%.10f' % (
                (epoch + 1), epoch_loss, epoch_seg_loss, epoch_grad_loss, epoch_val_loss, epoch_seg_val_loss,
                epoch_grad_val_loss)
        )

    # Save losses and model
    torch.save(reg_model.state_dict(), os.path.join(RESULT_FOLDER, 'model.pth'))
    save_losses(losses, val_losses)

    # Free some memory
    del dataloader, dataloader_val, dataset, dataset_val, X_train, y_train, X_val, y_val
    gc.collect()

    return reg_model


def evaluate_network(reg_model):
    _, _, (X_test, y_test) = read_data(
        visceral_path='./visceral_gold_1.5mm.h5',
        image_shape=image_shape,
        split=[0, 0, 20]
    )

    assert len(X_test) == 20
    assert len(y_test) == 20

    report_support_recon_loss_organs(X_test, y_test, cfg['test_classes'], cfg['shots'], reg_model, device)
    report_few_shot_test_organs(X_test, y_test, reg_model, cfg['shots'], device)


if __name__ == "__main__":
    logging.info("Device: {}".format(device))

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nb_epochs', type=int, required=True, help='Number of Epochs')
    parser.add_argument('--diff_min', type=int, required=True, help='Minimum diff for choosing two slices')
    parser.add_argument('--diff_max', type=int, required=True, help='Maximum diff for choosing two slices')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size')
    parser.add_argument('--steps_per_epoch', type=int, required=True, help='Steps per epoch')

    parser.add_argument('--sim_loss_type', type=str, required=True, help='Similarity Loss Type (mes or ncc)')
    parser.add_argument('--use_smooth_loss', type=str2bool, required=True, help='Weather to use smooth loss or not')
    parser.add_argument('--use_seg_loss', type=str2bool, required=True, help='Weather to use segmentation loss or not')

    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter (for smooth loss)')
    parser.add_argument('--lamda', type=float, required=True, help='Lambda parameter (for segmentation loss)')

    parser.add_argument('--test_classes', type=int, nargs='*', required=True, help='Test classes')
    parser.add_argument('--training_classes', type=int, nargs='*', required=True, help='Training classes')
    parser.add_argument('--val_classes', type=int, nargs='*', required=True, help='Validation classes')

    parser.add_argument('--shots', type=int, required=True, help='Number of Training Shots')

    args = parser.parse_args()

    new_cfg = {
        'nb_epochs': args.nb_epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'batch_size': args.batch_size,
        'diff_min': args.diff_min,
        'diff_max': args.diff_max,

        'use_smooth_loss': args.use_smooth_loss,
        'use_seg_loss': args.use_seg_loss,
        'sim_loss_type': args.sim_loss_type,
        'gamma': args.gamma,
        'lamda': args.lamda,

        'training_classes': args.training_classes,
        'val_classes': args.val_classes,
        'test_classes': args.test_classes,

        'shots': args.shots,
    }

    assert set(args.training_classes + args.test_classes + args.val_classes) == set(range(1, 7))
    assert len(args.training_classes) + len(args.test_classes) + len(args.val_classes) == 6

    set_cfg(new_cfg)

    logging.info("Config:")

    logging.info(cfg)

    # Train the network
    reg_model = train_network()

    # Evaluate on test data
    evaluate_network(reg_model)

    # Download the reports and model
    download_results()
