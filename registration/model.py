import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# the 2d unet module used by voxelmorph
# this module is converted from tf/keras to pytorch, line by line.
class UNet(nn.Module):
    def __init__(self, ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1):
        super(UNet, self).__init__()

        # the number of dimensions of the images. 2 in our case.
        self.ndims = ndims

        # the shape of the images. image_shape in our case.
        self.vol_size = vol_size

        # list of features for encoder layers.
        self.enc_nf = enc_nf

        # list of features for decoder layers.
        self.dec_nf = dec_nf

        # works only if the dec_nf has a length of at least 6
        self.full_size = full_size

        # src is the moving image
        # tgt is the fixed image
        # src_feats is the number of features (channels) of src. 1 in our case.
        # same goes for tgt_feats. but is case of RNN, tgt_feats will be zero.
        self.src_feats = src_feats
        self.tgt_feats = tgt_feats

        # <---- ENCODER ---->

        # initialize encoder convolutional layers.
        self.down_convs = nn.ModuleList()

        # we concat the src and tgt. so the input to the first convolution will have src_feats + tgt_feats features.
        # also, `features` always keeps track of the number of output features of the last layer initialized.
        features = self.src_feats + self.tgt_feats

        for i in range(len(self.enc_nf)):
            # create encoder conv
            self.down_convs.append(self.conv_block(self.ndims, features, self.enc_nf[i], 2))

            # update `features`
            features = self.enc_nf[i]

        # <---- ENCODER ---->

        # <---- DECODER ---->
        self.up_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(5):
            # convolution
            self.up_convs.append(self.conv_block(self.ndims, features, self.dec_nf[i]))

            # update `features`
            features = self.dec_nf[i]

            if i < 3:  # we have upsamples in the first three
                w = int(vol_size[0] / 2 ** (3 - i))
                h = int(vol_size[1] / 2 ** (3 - i))
                self.upsamples.append(self.upsample_layer(size=(w, h)))

                # update `features`
                features += self.enc_nf[-i - 2]
        # <---- DECODER ---->

        if self.full_size:
            self.fs_upsample = self.upsample_layer(size=vol_size)

            # the result of this upsample will be concated with both src and tgt.
            features += self.src_feats + self.tgt_feats

            # conv
            self.fs_conv = self.conv_block(self.ndims, features, self.dec_nf[5])
            features = self.dec_nf[5]

        if len(self.dec_nf) == 7:  # this is for voxel-morph-2. not used in our code.
            self.extra_conv = self.conv_block(self.ndims, features, self.dec_nf[6])
            features = self.dec_nf[6]

        # store the number of output features
        self.out_features = features

    def forward(self, src, tgt=None):
        # shape of src and tgt: (batch_size, src_feats, *vol_size)

        # concat src and tgt
        x = src
        if tgt is not None:
            x = torch.cat([x, tgt], dim=1)

        # encoder
        x_enc = [x]
        for i in range(len(self.enc_nf)):
            x = self.down_convs[i](x)
            x_enc.append(x)

        # decoder
        for i in range(5):
            x = self.up_convs[i](x)
            if i < 3:
                x = self.upsamples[i](x)
                x = torch.cat([x, x_enc[-i - 2]], dim=1)

        # full_size
        if self.full_size:
            x = self.fs_upsample(x)
            x = torch.cat([x, x_enc[0]], dim=1)
            x = self.fs_conv(x)

        # extra
        if len(self.dec_nf) == 7:
            x = self.extra_conv(x)

        return x

    def upsample_layer(self, size):
        return nn.Upsample(size=size)

    def conv_block(self, ndims, in_features, out_features, strides=1):
        Conv = getattr(nn, 'Conv%dd' % ndims)
        conv = Conv(in_features, out_features, kernel_size=3, padding=1, stride=strides)
        nn.init.xavier_normal_(conv.weight)
        return nn.Sequential(conv, nn.LeakyReLU(.2, False))


# Copied from the internet for spatial transoform
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


# a model to get two moving and fixed images and find the transformation between them,
class RegModel(nn.Module):
    def __init__(self, ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1, mode='bilinear'):
        super(RegModel, self).__init__()

        # unet
        self.unet = UNet(ndims, vol_size, enc_nf, dec_nf, full_size=True, src_feats=1, tgt_feats=1)

        # displacement conv
        self.disp_layer = nn.Conv2d(dec_nf[-1], ndims, kernel_size=3, padding=1)

        # transformation
        self.spatial_transformer = SpatialTransformer(size=vol_size, mode=mode)

    def forward(self, src, tgt):
        # pass through unet
        unet_out = self.unet(src, tgt)

        # find the transformation
        disp = self.disp_layer(unet_out)

        # use transformation to regenerate tgt
        moved = self.spatial_transform(src, disp)

        # return both transformed image and the transfromation itself
        return moved, disp

    def spatial_transform(self, src, disp):
        return self.spatial_transformer(src, disp)
