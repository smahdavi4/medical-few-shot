import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.losses import get_recon_loss
from utils.registration import get_moved_image, get_moved_seg
from config import cfg
from utils.colab import RESULT_FOLDER


def dice_score_perclass(vol_output, ground_truth, num_classes):
    dice_perclass = torch.zeros(num_classes)
    for i in range(num_classes):
        gt = (ground_truth == i).float()
        pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(gt, pred))
        union = torch.sum(gt) + torch.sum(pred) + 0.00001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


# 0:Background 1:Liver, 2:Spleen 3:LKidney 4:RKidney 5:LPsoas 6:RPsoas
def report_all_organs(X, y, reg_model, device):
    f = open('visceral.log', 'w')
    for m_pat in range(0, len(X)):
        dices = []
        for f_pat in range(0, len(X)):
            if m_pat == f_pat:
                continue
            pred_f = np.zeros(X[f_pat].shape)
            for f_slice in range(len(X[f_pat])):
                m_slice = int(len(X[m_pat]) / len(X[f_pat]) * f_slice)
                fixed = X[f_pat][f_slice, np.newaxis, ...]
                fixed_seg = torch.tensor(y[f_pat][f_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
                moving = X[m_pat][m_slice, np.newaxis, ...]
                moving_seg = torch.tensor(y[m_pat][m_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
                moved, disp = get_moved_image(reg_model, fixed, moving, device)

                img_pred = get_moved_seg(reg_model, disp, moving_seg, device).cpu().numpy()[0, 0, ...]

                pred_f[f_slice] = img_pred
            dice_score = dice_score_perclass(torch.tensor(pred_f).to(device), torch.tensor(y[f_pat]).to(device), 7)
            f.write(
                "Segmented Patient:" + str(m_pat) + " Result on Patient " + str(f_pat) + ": " + str(dice_score) + "\n")
            dices.append(dice_score)
        f.write(
            "Mean Dice for Segmented Patient " + str(m_pat) + " " + str(torch.mean(torch.stack(dices), dim=0)) + "\n")
        mean_dice = torch.mean(torch.stack(dices), dim=0).detach().cpu().numpy()
        print("Mean Dice for Segmented Patient ",
              (mean_dice[1] + mean_dice[2] + (mean_dice[3] + mean_dice[4]) / 2 + (mean_dice[5] + mean_dice[6]) / 2) / 4)
        f.write("-" * 10 + "\n")
        print(m_pat)
    f.close()


# 0:Background 1:Liver, 2:Spleen 3:LKidney 4:RKidney 5:LPsoas 6:RPsoas
def report_test_organs(X, y, reg_model, device):
    logging.info("-" * 20)
    logging.info("Training classes: " + str(cfg['training_classes']) + " Testing Classes: " + str(cfg['test_classes']))
    for m_pat in range(0, len(X)):
        dices = []
        for f_pat in range(0, len(X)):
            if m_pat == f_pat:
                continue
            pred_f = np.zeros(X[f_pat].shape)

            for f_slice in range(len(X[f_pat])):
                m_slice = int(len(X[m_pat]) / len(X[f_pat]) * f_slice)
                fixed = X[f_pat][f_slice, np.newaxis, ...]
                fixed_seg = torch.tensor(y[f_pat][f_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
                moving = X[m_pat][m_slice, np.newaxis, ...]
                moving_seg = torch.tensor(y[m_pat][m_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
                moved, disp = get_moved_image(reg_model, fixed, moving, device)

                img_pred = get_moved_seg(reg_model, disp, moving_seg, device).cpu().numpy()[0, 0, ...]
                img_true = fixed_seg.detach().cpu().numpy()[0, 0, ...]

                # Only the slices that contain the organ
                true_clss = np.unique(img_true)
                for cls in range(7):
                    if cls not in true_clss:
                        img_pred[img_pred == cls] = 0
                    pred_f[f_slice] = img_pred
            dice_score = dice_score_perclass(torch.tensor(pred_f).to(device), torch.tensor(y[f_pat]).to(device), 7)
            logging.info(
                "Segmented Patient:" + str(m_pat) + " Result on Patient " + str(f_pat) + ": " + str(dice_score))
            dices.append(dice_score)

        mean_dice = torch.mean(torch.stack(dices), dim=0).detach().cpu().numpy()
        for cls in cfg['test_classes']:
            logging.info(
                "Mean Dice for Segmented Patient " + str(m_pat) + " Organ: " + str(cls) + " " + str(mean_dice[cls]))
        logging.info("-" * 20)


################### Choosing Support volume

def report_support_recon_loss_whole_volume(X_train, X_test, reg_model, device):
    logging.info("-" * 20)

    recon_loss = get_recon_loss()

    for m_pat in range(len(X_test)):  # Support
        recon_losses = []
        for f_pat in range(len(X_train)):  # Query
            f_pat_recon_losses = []
            for f_slice in range(X_train[f_pat].shape[0]):
                m_slice = int(len(X_test[m_pat]) / len(X_train[f_pat]) * f_slice)

                fixed = X_train[f_pat][f_slice, np.newaxis, ...]
                moving = X_test[m_pat][m_slice, np.newaxis, ...]
                moved, disp = get_moved_image(reg_model, fixed, moving, device)

                fixed_torch = torch.tensor(fixed).to(device).float().unsqueeze(0)

                f_pat_recon_losses.append(recon_loss(moved, fixed_torch).item())
            recon_losses.append(np.mean(f_pat_recon_losses))
        mean_recon_loss = np.mean(recon_losses)
        var_recon_loss = np.var(recon_losses)
        logging.info("Segmented Patient {}: Recon loss mean: {} , Recon loss var: {}".format(
            m_pat, mean_recon_loss, var_recon_loss
        ))
    logging.info("-" * 20)


def report_support_recon_loss_organs(X, y, organ_ids, shots, reg_model, device):
    logging.info("-" * 20)

    start_organs_seg, end_organs_seg = _get_start_end_organs(y)
    recon_loss = get_recon_loss()

    for m_pat in range(len(X)):  # Support
        recon_losses = []
        for f_pat in range(0, len(X)):  # Query
            if m_pat == f_pat:
                continue
            f_pat_recon_losses = []
            for organ_cls in organ_ids:
                f_pat_organ_losses = []
                for f_slice in range(start_organs_seg[f_pat, organ_cls], end_organs_seg[f_pat, organ_cls] + 1):  # Query
                    m_slice = _get_fewshot_support_slice(
                        start_organs_seg[m_pat, organ_cls],
                        end_organs_seg[m_pat, organ_cls],
                        start_organs_seg[f_pat, organ_cls],
                        end_organs_seg[f_pat, organ_cls], f_slice, shots
                    )

                    fixed = X[f_pat][f_slice, np.newaxis, ...]
                    moving = X[m_pat][m_slice, np.newaxis, ...]
                    moved, disp = get_moved_image(reg_model, fixed, moving, device)

                    fixed_torch = torch.tensor(fixed).to(device).float().unsqueeze(0)

                    f_pat_organ_losses.append(recon_loss(moved, fixed_torch).item())
                f_pat_recon_losses.append(np.mean(f_pat_organ_losses))
            recon_losses.append(np.mean(f_pat_recon_losses))
        mean_recon_loss = np.mean(recon_losses)
        var_recon_loss = np.var(recon_losses)
        logging.info("Segmented Patient {}: Recon loss mean: {} , Recon loss var: {}".format(
            m_pat, mean_recon_loss, var_recon_loss
        ))
    logging.info("-" * 20)


################### Fewshot setting:

def _get_start_end_organs(y):
    organs = 7
    N = len(y)
    start = -1 * np.ones((N, organs), dtype=np.int)
    end = -1 * np.ones((N, organs), dtype=np.int)

    for patient in range(N):
        for slc in range(y[patient].shape[0]):
            for org_id in np.unique(y[patient][slc]):
                if org_id > 6:
                    logging.error("Organ Id Larger than 6. Skipping")
                    continue
                if start[patient, org_id] == -1:
                    start[patient, org_id] = slc
                end[patient, org_id] = slc

    return start, end


def _get_fewshot_support_slice(support_start, support_end, query_start, query_end, query_idx, shots):
    support_len = support_end - support_start + 1
    query_len = query_end - query_start + 1
    shots = min(shots, support_len)
    support_idx = int((query_idx - query_start) / query_len * support_len) + support_start
    support_items = support_start + np.round(np.linspace(0, support_len - 1, 2 * shots + 1)).astype(int)[
                                    1::2]  # the slices for which we have the segmentation
    nearest_support_item = support_items[(np.abs(support_items - support_idx)).argmin()]
    return nearest_support_item


def report_few_shot_test_organs(X, y, reg_model, shots, device):
    logging.info("-" * 20)
    logging.info("Training classes: " + str(cfg['training_classes']) + " Testing Classes: " + str(cfg['test_classes']))

    start_organs_seg, end_organs_seg = _get_start_end_organs(y)

    for m_pat in range(0, len(X)):  # Support
        dices = []
        for f_pat in range(0, len(X)):  # Query
            if m_pat == f_pat:
                continue
            pred_f = np.zeros(X[f_pat].shape)

            for organ_cls in range(1, 7):
                for f_slice in range(start_organs_seg[f_pat, organ_cls], end_organs_seg[f_pat, organ_cls] + 1):
                    m_slice = _get_fewshot_support_slice(
                        start_organs_seg[m_pat, organ_cls],
                        end_organs_seg[m_pat, organ_cls],
                        start_organs_seg[f_pat, organ_cls],
                        end_organs_seg[f_pat, organ_cls], f_slice, shots
                    )

                    fixed = X[f_pat][f_slice, np.newaxis, ...]
                    moving = X[m_pat][m_slice, np.newaxis, ...]
                    moving_seg = torch.tensor(y[m_pat][m_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
                    moved, disp = get_moved_image(reg_model, fixed, moving, device)

                    img_pred = get_moved_seg(reg_model, disp, moving_seg, device).cpu().numpy()[0, 0, ...]

                    # TODO: FIX Hack (Some border pixels get assigned to organs!)
                    try:
                        assert np.sum(img_pred[:5, :]) == 0
                        assert np.sum(img_pred[:, :5]) == 0
                        assert np.sum(img_pred[250:, :]) == 0
                        assert np.sum(img_pred[:, 250:]) == 0
                    except:
                        logging.error("Boundary pixel assigned to a non-background class.")
                        plt.imshow(img_pred)
                        plt.show()

                    pred_f[f_slice][img_pred == organ_cls] = img_pred[img_pred == organ_cls]
                    # pred_f[f_slice][y[m_pat][m_slice] == organ_cls] = y[m_pat][m_slice][y[m_pat][m_slice] == organ_cls]

            dice_score = dice_score_perclass(torch.tensor(pred_f).to(device), torch.tensor(y[f_pat]).to(device), 7)
            logging.info(
                "Segmented Patient:" + str(m_pat) + " Result on Patient " + str(f_pat) + ": " + str(dice_score))
            dices.append(dice_score)

        mean_dice = torch.mean(torch.stack(dices), dim=0).detach().cpu().numpy()
        for cls in cfg['test_classes']:
            logging.info(
                "Mean Dice for Segmented Patient " + str(m_pat) + " Organ: " + str(cls) + " " + str(mean_dice[cls]))
        logging.info("-" * 20)


def visualize_few_shot_test_organs(X, y, support_pat_id, query_pat_id, organ_cls, reg_model, shots, device):
    logging.info("-" * 20)
    logging.info("support_pat_id: {}, query_pat_id: {}".format(support_pat_id, query_pat_id))

    start_organs_seg, end_organs_seg = _get_start_end_organs(y)

    m_pat = support_pat_id
    f_pat = query_pat_id

    pred_f = np.zeros(X[f_pat].shape)

    for f_slice in range(start_organs_seg[f_pat, organ_cls], end_organs_seg[f_pat, organ_cls] + 1):
        m_slice = _get_fewshot_support_slice(
            start_organs_seg[m_pat, organ_cls],
            end_organs_seg[m_pat, organ_cls],
            start_organs_seg[f_pat, organ_cls],
            end_organs_seg[f_pat, organ_cls], f_slice, shots
        )

        fixed = X[f_pat][f_slice, np.newaxis, ...]
        fixed_seg = torch.tensor(y[f_pat][f_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
        moving = X[m_pat][m_slice, np.newaxis, ...]
        moving_seg = torch.tensor(y[m_pat][m_slice]).to(device).float().unsqueeze(0).unsqueeze(0)
        moved, disp = get_moved_image(reg_model, fixed, moving, device)

        img_pred = get_moved_seg(reg_model, disp, moving_seg, device).cpu().numpy()[0, 0, ...]
        img_true = fixed_seg.detach().cpu().numpy()[0, 0, ...]

        if f_slice % 5 == 0:
            logging.info("Query Slice: {}, Support Slice: {}".format(f_slice, m_slice))
            print("Query Slice: {}, Support Slice: {}".format(f_slice, m_slice))
            plt.figure(figsize=(31, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(X[m_pat][m_slice], 'gray', interpolation='none')
            plt.imshow(
                moving_seg.detach().cpu().numpy()[0, 0, ...] == organ_cls, 'jet', interpolation='none',
                alpha=0.1
            )
            plt.title("Support", fontdict={'fontsize': 40})
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(X[f_pat][f_slice], 'gray', interpolation='none')
            plt.imshow(img_true == organ_cls, 'jet', interpolation='none', alpha=0.1)
            plt.title("Query Ground Truth", fontdict={'fontsize': 40})
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(X[f_pat][f_slice], 'gray', interpolation='none')
            plt.imshow(img_pred == organ_cls, 'jet', interpolation='none', alpha=0.1)
            plt.title("Query Prediction", fontdict={'fontsize': 40})
            plt.axis('off')
            #
            # plt.subplot(1, 6, 4)
            # plt.imshow(img_pred == organ_cls, 'gray', interpolation='none')
            # plt.imshow(img_true == organ_cls, 'jet', interpolation='none', alpha=0.3)
            # plt.axis('off')
            #
            # plt.subplot(1, 6, 5)
            # plt.imshow(y[m_pat][m_slice] == organ_cls, 'gray', interpolation='none')
            # plt.imshow(img_pred == organ_cls, 'jet', interpolation='none', alpha=0.3)
            # plt.axis('off')
            #
            # plt.subplot(1, 6, 6)
            # plt.imshow(y[m_pat][m_slice] == organ_cls, 'gray', interpolation='none')
            # plt.imshow(img_true == organ_cls, 'jet', interpolation='none', alpha=0.3)
            # plt.axis('off')
            plt.savefig('{}.pdf'.format(f_slice))
            plt.show()

        pred_f[f_slice][img_pred == organ_cls] = img_pred[img_pred == organ_cls]
    dice_score = dice_score_perclass(torch.tensor(pred_f).to(device), torch.tensor(y[f_pat]).to(device), 7)
    print("Dice: ", dice_score)
    logging.info("-" * 20)


####################

def save_losses(losses_train, losses_val):
    plt.plot(np.arange(len(losses_train)), losses_train, '.-', label='Train')
    plt.plot(np.arange(len(losses_val)), losses_val, '.-', label='Val')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(RESULT_FOLDER, 'loss.png'))
    plt.clf()
