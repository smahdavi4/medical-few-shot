import numpy as np
import PIL
from PIL import Image

import torchvision.transforms.functional as tr_F
from torchvision.transforms import Pad, CenterCrop

def to_scale(img, shape=None):

    height, width = shape
    if img.dtype == np.uint8:
        return np.array(Image.fromarray(img).resize((height,width), PIL.Image.NEAREST)).astype(np.uint8)
    elif img.dtype == np.float:
        max_ = np.max(img)
        factor = 255.0/max_ if max_ != 0 else 1
        return (np.array(Image.fromarray(img).resize((height,width), PIL.Image.NEAREST)) / factor).astype(np.float)
    else:
        raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')

def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


class CropOrPad(object):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size
    
    def __call__(self, img):
        w, h = img.size
        H, W = self.out_size
        if w >= W and h >= H:
            return tr_F.center_crop(img, self.out_size)
        else:
            top_pad = int(np.floor((H - h)  / 2))
            buttom_pad = int(np.ceil((H - h)  / 2))
            left_pad = int(np.floor((W - w)  / 2))
            right_pad = int(np.ceil((W - w)  / 2))
            return tr_F.pad(img, (left_pad, top_pad, right_pad, buttom_pad), fill=0, padding_mode='symmetric')
