import numpy as np
import torch


class InterPatientDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, diff_min, diff_max, steps_per_epoch, batch_size):
        self.X = X
        self.y = y
        self.diff_min = diff_min
        self.diff_max = diff_max
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def __getitem__(self, index):
        idx_3d_sample = np.random.randint(0, len(self.X))
        idx2_3d_sample = np.random.randint(0, len(self.X))

        image_3d_sample, seg_3d_sample = self.X[idx_3d_sample], self.y[idx_3d_sample]
        image2_3d_sample, seg2_3d_sample = self.X[idx2_3d_sample], self.y[idx2_3d_sample]

        diff_slices = np.random.randint(self.diff_min, self.diff_max)
        idx1 = np.random.randint(0, image_3d_sample.shape[0] - diff_slices)
        idx2 = int(image2_3d_sample.shape[0] / image_3d_sample.shape[0] * (idx1 + diff_slices))

        if np.random.randint(0, 2) == 0:
            fixed_image = image_3d_sample[idx1, np.newaxis, ...]
            fixed_seg = seg_3d_sample[idx1, np.newaxis, ...].astype(np.int32)
            moving_image = image2_3d_sample[idx2, np.newaxis, ...]
            moving_seg = seg2_3d_sample[idx2, np.newaxis, ...].astype(np.int32)
        else:
            fixed_image = image2_3d_sample[idx2, np.newaxis, ...]
            fixed_seg = seg2_3d_sample[idx2, np.newaxis, ...].astype(np.int32)
            moving_image = image_3d_sample[idx1, np.newaxis, ...]
            moving_seg = seg_3d_sample[idx1, np.newaxis, ...].astype(np.int32)

        return moving_image, moving_seg, fixed_image, fixed_seg
