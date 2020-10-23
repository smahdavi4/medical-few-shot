import torch
import matplotlib.pyplot as plt


def get_moved_image(reg_model, fixed, moving, device, verbose=False):
    moving = torch.tensor(moving).to(device).float().unsqueeze(0)
    fixed = torch.tensor(fixed).to(device).float().unsqueeze(0)

    moved, disp = reg_model(moving, fixed)

    if verbose:
        plt.subplot(1, 4, 1)
        plt.imshow(moving[0, 0, :, :].detach().cpu().numpy())
        plt.subplot(1, 4, 2)
        plt.imshow(fixed[0, 0, :, :].detach().cpu().numpy())

        plt.subplot(1, 4, 3)
        plt.imshow(moved[0, 0, :, :].detach().cpu().numpy())

        plt.subplot(1, 4, 4)
        plt.imshow(disp[0, 0, :, :].detach().cpu().numpy())

        plt.show()
    return moved, disp


def get_moved_seg(reg_model, disp, moving_seg, device, verbose=False):
    # moved_seg = torch.zeros([7] + list(moving_seg.shape)).to(device)
    moved_seg = torch.zeros_like(moving_seg)
    for cls in range(1, 7):
        moving_seg_class = (moving_seg == cls).to(device).float()
        moved_seg_cls = reg_model.spatial_transform(moving_seg_class, disp).detach()
        moved_seg[moved_seg_cls > 0.5] = cls
    return moved_seg


def get_moved_seg__(reg_model, disp, moving_seg, device, verbose=False):
    moved_seg = torch.zeros([7] + list(moving_seg.shape)).to(device)
    for cls in range(0, 7):
        moving_seg_class = (moving_seg == cls).to(device).float()
        moved_seg[cls] = reg_model.spatial_transform(moving_seg_class, disp).detach()
    return torch.argmax(moved_seg, dim=0)
