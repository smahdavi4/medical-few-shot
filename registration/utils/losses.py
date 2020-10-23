import torch
import torch.nn.functional as F
import numpy as np
import math

from config import cfg


def get_recon_loss():
    return torch.nn.MSELoss() if cfg['sim_loss_type'] == "mse" else ncc_loss


# Copied from voxed morph's github code
def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims == 2

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    stride = (1, 1)
    padding = (pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


############

def gradient_loss(s, penalty='l2'):
    assert len(s.shape) == 4

    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


###########

def binary_dice_loss(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 4
    # assert len(torch.unique(y_pred)) == 2
    # assert len(torch.unique(y_true)) == 2

    numerator = 2 * torch.sum(y_true * y_pred, dim=[1, 2, 3])
    denominator = torch.clamp(torch.sum(y_true + y_pred, [1, 2, 3]), min=1e-5)
    dice = torch.mean(torch.true_divide(numerator, denominator))
    return -dice
