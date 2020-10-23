# Adapted from https://github.com/multimodallearning/miccai19_self_supervision/blob/master/miccai_train.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F

# network and 

# defining the D2D-CNN of the paper used to learn inherent anatomical knowledge just by 
# spatial relations;
# this part will serve as the feature extractor:
# it takes (3,42,42) - (chan,x,y) inputs, meaning: along the current axis, 3 neighboring planes with 
# spatial extensions (42,42) are processed channelwise;
# after being processed by this architecture, the output is of spatial size (1,1) and has 64 channels
            
class ConvNet(nn.Module):
    def __init__(self):
        # takes 3 neighboring slices as channels
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 3, stride=1, dilation=1, bias=False, groups=1)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = torch.nn.GroupNorm(4,32)
        self.acti1 = torch.nn.LeakyReLU()
        
        self.layer2 = nn.Conv2d(32, 32, 3, stride=1, dilation=1, bias=False, groups=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm2 = torch.nn.GroupNorm(4,32)
        self.acti2 = torch.nn.LeakyReLU()
        
        self.layer3 = nn.Conv2d(32, 64, 3, stride=1, dilation=1, bias=False, groups=1)
        self.norm3 = torch.nn.GroupNorm(4,64)
        self.acti3 = torch.nn.LeakyReLU()
        
        self.layer4 = nn.Conv2d(64, 64, 3, stride=1, dilation=1, bias=False, groups=1)
        self.norm4 = torch.nn.GroupNorm(4,64)
        self.acti4 = torch.nn.LeakyReLU()
        
        self.layer5 = nn.Conv2d(64, 64, 3, stride=1, dilation=1, bias=False, groups=1)
        self.norm5 = torch.nn.GroupNorm(4,64)
        self.acti5 = torch.nn.LeakyReLU()
        
        self.layer6 = nn.Conv2d(64, 64, 3, stride=1, dilation=1, bias=False, groups=1)
        self.norm6 = torch.nn.GroupNorm(4,64)
        self.acti6 = torch.nn.LeakyReLU()
        
    def forward(self, patches):
        x = self.acti1(self.norm1(self.mp1(self.layer1(patches))))
        x = self.acti2(self.norm2(self.mp2(self.layer2(x))))
        x = self.acti3(self.norm3(self.layer3(x)))
        x = self.acti4(self.norm4(self.layer4(x)))
        x = self.acti5(self.norm5(self.layer5(x)))
        x = self.acti6(self.norm6(self.layer6(x)))
        x_feat = x
        return x_feat


# this CNN is used during training, when we apply our proposed trainig scheme;
# i.e. in a siamese-fashion, we process two patches with the D2D-CNN above to
# generate feature decriptors for these patches and subsequently, we pass both
# descriptors to this "HeatNet"; here, from the concatenated feature representations
# of size (128,1,1) this network will be trained to retrieve a spatial heatmap
# of size (1,19,19) -> the groundtruth is generated with the function "heatmap_gen" defined above
class HeatNet(nn.Module):
    def __init__(self):

        super(HeatNet, self).__init__()
        self.layer1 = nn.Conv2d(128, 64, 1, bias=False, groups=1)
        self.norm1 = torch.nn.GroupNorm(4,64)
        self.acti1 = torch.nn.LeakyReLU()
        
        self.layer2 = nn.Conv2d(64, 32, 1, bias=False, groups=1)
        self.norm2 = torch.nn.GroupNorm(4,32)
        self.acti2 = torch.nn.LeakyReLU()
        
        self.layer2_a = nn.Conv2d(32, 16, 1, bias=False, groups=1)
        self.norm2_a = torch.nn.GroupNorm(4,16)
        self.acti2_a = torch.nn.LeakyReLU()
        
        self.layer3_0 = nn.ConvTranspose2d(16, 16, 5, bias=False, groups=1)
        self.norm3_0 = torch.nn.GroupNorm(4,16)
        self.acti3_0 = torch.nn.LeakyReLU()
        self.layer3_1 = nn.Conv2d(16, 16, 3, bias=False, groups=1)
        self.norm3_1 = nn.BatchNorm2d(16)
        self.acti3_1 = torch.nn.LeakyReLU()
        
        self.layer4_0 = nn.ConvTranspose2d(16, 16, 5, bias=False, groups=1)
        self.norm4_0 = torch.nn.GroupNorm(4,16)
        self.acti4_0 = torch.nn.LeakyReLU()
        self.layer4_1 = nn.Conv2d(16, 8, 3, bias=False, groups=1)
        self.norm4_1 = nn.BatchNorm2d(8)
        self.acti4_1 = torch.nn.LeakyReLU()
        
        self.layer5_0 = nn.ConvTranspose2d(8, 4, 5, bias=False, groups=1)
        self.norm5_0 = torch.nn.GroupNorm(4,4)
        self.acti5_0 = torch.nn.LeakyReLU()
        self.layer5_1 = nn.Conv2d(4, 1, 3, padding=1, groups=1)

    def forward(self, x_feat1, x_feat2):

        x = self.acti1(self.norm1(self.layer1(torch.cat((x_feat1,x_feat2),1))))
        x = self.acti2(self.norm2(self.layer2(x)))
        x = self.acti2_a(self.norm2_a(self.layer2_a(x)))
        
        x = self.acti3_0(self.norm3_0(self.layer3_0(x)))
        x = self.acti3_1(self.norm3_1(self.layer3_1(x)))
        
        x = F.interpolate(x,size=(11,11),mode='bilinear',align_corners=True)
        
        x = self.acti4_0(self.norm4_0(self.layer4_0(x)))
        x = F.avg_pool2d(self.acti4_1(self.norm4_1(self.layer4_1(x))),3,padding=1,stride=1)

        x = self.acti5_0(self.norm5_0(self.layer5_0(x)))
        
        x = F.interpolate(x,size=(19,19),mode='bilinear',align_corners=True)
        
        heatmap = self.layer5_1(x)
        return heatmap
    
# this CNN is used during training, when we apply our proposed trainig scheme;
# i.e. in a siamese-fashion, we process two patches with the D2D-CNN above to
# generate feature decriptors for these patches and subsequently, we pass both
# descriptors to this "OffNet"; here, from the concatenated feature representations
# of size (128,1,1) this network will be trained to just output the two offset parameters
# that define the inplane displacement in contrast to the spatial reconstruction of the 
# HeatMap approach
class OffNet(nn.Module):
    def __init__(self):

        super(OffNet, self).__init__()
        self.layer1 = nn.Conv2d(128, 128, 1, bias=False, groups=1)
        self.norm1 = torch.nn.GroupNorm(4,128)
        self.acti1 = torch.nn.LeakyReLU()
        
        self.layer2 = nn.Conv2d(128, 64, 1, bias=False, groups=1)
        self.norm2 = torch.nn.GroupNorm(4,64)
        self.acti2 = torch.nn.LeakyReLU()
        
        self.layer3 = nn.Conv2d(64, 32, 1, bias=False, groups=1)
        self.norm3 = torch.nn.GroupNorm(4,32)
        self.acti3 = torch.nn.LeakyReLU()
        
        self.layer_out = nn.Conv2d(32, 2, 1, groups=1)

    def forward(self, x_feat1, x_feat2):

        x = self.acti1(self.norm1(self.layer1(torch.cat((x_feat1,x_feat2),1))))
        x = self.acti2(self.norm2(self.layer2(x)))
        x = self.acti3(self.norm3(self.layer3(x)))
        
        off_pred = self.layer_out(x)
        return off_pred

# this CNN is the descriptor part of the 3D extended Doersch approach;
# cubes of inputsize (25,25,25) will be turned into descriptors of size 1^3
# and 192 channels
class DoerschNet(nn.Module):
    def __init__(self):

        super(DoerschNet, self).__init__()
        self.layer1 = nn.Conv3d(1, 16, 5, bias=False, groups=1)
        self.norm1 = torch.nn.GroupNorm(4,16)
        self.acti1 = torch.nn.LeakyReLU()
        
        self.layer2 = nn.Conv3d(16, 32, 3, dilation=2, bias=False, groups=1)
        self.norm2 = torch.nn.GroupNorm(4,32)
        self.acti2 = torch.nn.LeakyReLU()
        
        self.layer3 = nn.Conv3d(32, 32, 3, dilation=2, bias=False, groups=1)
        self.norm3 = torch.nn.GroupNorm(4,32)
        self.acti3 = torch.nn.LeakyReLU()
        
        self.layer4 = nn.Conv3d(32, 32, 3, dilation=2, bias=False, groups=1)
        self.norm4 = torch.nn.GroupNorm(4,32)
        self.acti4 = torch.nn.LeakyReLU()
        
        self.layer5 = nn.Conv3d(32, 32, 3, dilation=1, bias=False, groups=1)
        self.norm5 = torch.nn.GroupNorm(4,32)
        self.acti5 = torch.nn.LeakyReLU()
        
        self.layer6 = nn.Conv3d(32, 32, 5, bias=False, groups=1)
        self.norm6 = torch.nn.GroupNorm(4,32)
        self.acti6 = torch.nn.LeakyReLU()
        
        self.layer7 = nn.Conv3d(32, 3*64, 3, bias=False, groups=1)
        self.norm7 = torch.nn.GroupNorm(4,3*64)
        self.acti7 = torch.nn.LeakyReLU()
        

    def forward(self, x):

        x = self.acti1(self.norm1(self.layer1(x)))
        x = self.acti2(self.norm2(self.layer2(x)))
        x = self.acti3(self.norm3(self.layer3(x)))
        x = self.acti4(self.norm4(self.layer4(x)))
        x = self.acti5(self.norm5(self.layer5(x)))
        x = self.acti6(self.norm6(self.layer6(x)))
        x = self.acti7(self.norm7(self.layer7(x)))
        
        return x
       
# similar to the offset parameter regression approach, this decoder part also
# takes as input two feature encodings by the DoerschCNN in a siamese manner.
# here, the auxiliary task is a classification problem instead of a regression task;
# the output encodes the relative position of encoding1 with respect to encoding2
# (top/bottom, left/right, front/back)
class DoerschDecodeNet(nn.Module):
    def __init__(self):

        super(DoerschDecodeNet, self).__init__()
        self.layer1 = nn.Conv3d(2*3*64, 64, 1, bias=False, groups=1)
        self.norm1 = torch.nn.GroupNorm(4,64)
        self.acti1 = torch.nn.LeakyReLU()
        
        self.layer2 = nn.Conv3d(64, 64, 1, bias=False, groups=1)
        self.norm2 = torch.nn.GroupNorm(4,64)
        self.acti2 = torch.nn.LeakyReLU()
        
        self.layer3 = nn.Conv3d(64, 32, 1, bias=False, groups=1)
        self.norm3 = torch.nn.GroupNorm(4,32)
        self.acti3 = torch.nn.LeakyReLU()
        
        self.layer_out = nn.Conv3d(32, 6, 1, groups=1)

    def forward(self, x_feat1, x_feat2):

        x = self.acti1(self.norm1(self.layer1(torch.cat((x_feat1,x_feat2),1))))
        x = self.acti2(self.norm2(self.layer2(x)))
        x = self.acti3(self.norm3(self.layer3(x)))
        
        neighbor_pred = self.layer_out(x)
        return neighbor_pred
