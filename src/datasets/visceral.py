import os
import logging
import random

import numpy as np
from PIL import Image
import h5py
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tr_F

from .base import FewShotDataset, MedicalFewshotDataset, MedicalNormalDataset
from utils.image import to_scale, normalize_image
from config import cfg


SILVER_PATIENT_LIST = [
    '10000101_1_CTce_ThAb',
    '10000114_1_CTce_ThAb',
    '10000115_1_CTce_ThAb',
    '10000116_1_CTce_ThAb',
    '10000117_1_CTce_ThAb',
    '10000118_1_CTce_ThAb',
    '10000119_1_CTce_ThAb',
    '10000120_1_CTce_ThAb',
    '10000121_1_CTce_ThAb',
    '10000122_1_CTce_ThAb',
    '10000124_1_CTce_ThAb',
    '10000125_1_CTce_ThAb',
    '10000126_1_CTce_ThAb',
    '10000150_1_CTce_ThAb',
    '10000151_1_CTce_ThAb',
    '10000152_1_CTce_ThAb',
    '10000153_1_CTce_ThAb',
    '10000154_1_CTce_ThAb',
    '10000155_1_CTce_ThAb',
    '10000156_1_CTce_ThAb',
    '10000158_1_CTce_ThAb',
    '10000159_1_CTce_ThAb',
    '10000160_1_CTce_ThAb',
    '10000161_1_CTce_ThAb',
    '10000162_1_CTce_ThAb',
    '10000163_1_CTce_ThAb',
    '10000164_1_CTce_ThAb',
    '10000165_1_CTce_ThAb',
    '10000166_1_CTce_ThAb',
    '10000167_1_CTce_ThAb',
    '10000168_1_CTce_ThAb',
    '10000169_1_CTce_ThAb',
    '10000170_1_CTce_ThAb',
    '10000171_1_CTce_ThAb',
    '10000172_1_CTce_ThAb',
    '10000173_1_CTce_ThAb',
    '10000174_1_CTce_ThAb',
    '10000175_1_CTce_ThAb',
    '10000177_1_CTce_ThAb',
    '10000178_1_CTce_ThAb',
    '10000179_1_CTce_ThAb',
    '10000180_1_CTce_ThAb',
    '10000181_1_CTce_ThAb',
    '10000183_1_CTce_ThAb',
    '10000184_1_CTce_ThAb',
    '10000185_1_CTce_ThAb',
    '10000186_1_CTce_ThAb',
    '10000187_1_CTce_ThAb',
    '10000188_1_CTce_ThAb',
    '10000189_1_CTce_ThAb',
    '10000190_1_CTce_ThAb',
    '10000191_1_CTce_ThAb',
    '10000192_1_CTce_ThAb',
    '10000193_1_CTce_ThAb',
    '10000194_1_CTce_ThAb',
    '10000196_1_CTce_ThAb',
    '10000198_1_CTce_ThAb',
    '10000199_1_CTce_ThAb',
    '10000200_1_CTce_ThAb',
    '10000201_1_CTce_ThAb',
    '10000203_1_CTce_ThAb',
    '10000204_1_CTce_ThAb',
    '10000205_1_CTce_ThAb'
]

PATIENT_LIST = [
    '10000110_1_CTce_ThAb',
    '10000100_1_CTce_ThAb',
    '10000130_1_CTce_ThAb',
    '10000133_1_CTce_ThAb',
    '10000113_1_CTce_ThAb',
    '10000108_1_CTce_ThAb',
    '10000105_1_CTce_ThAb',
    '10000128_1_CTce_ThAb',
    '10000135_1_CTce_ThAb',
    '10000136_1_CTce_ThAb',
    '10000106_1_CTce_ThAb',
    '10000127_1_CTce_ThAb',
    '10000109_1_CTce_ThAb',
    '10000104_1_CTce_ThAb',
    '10000134_1_CTce_ThAb',
    # '10000129_1_CTce_ThAb', Borken
    '10000132_1_CTce_ThAb',
    '10000112_1_CTce_ThAb',
    '10000111_1_CTce_ThAb',
    '10000131_1_CTce_ThAb'
]

organs = { # RadlexID
    'liver': {
        'id': 1,
        'radlex': ['58'],
    },
    'spleen': {
        'id': 2,
        'radlex': ['86'],
    },
    'lkidney': {
        'id': 3,
        'radlex': ['29662'],    
    },
    'rkidney': {
        'id': 4,
        'radlex': ['29663'],
    },
    'lpsoas_major': {
        'id': 5,
        'radlex': ['32248'],
    },
    'rpsoas_major': {
        'id': 6,
        'radlex': ['32249'],
    },
}

CLASS_WEIGHTS = { # Pre computed weights for classes, based on number of voxels of each class
    'background': 36771542,
    'liver': 1796158,
    'lungs': 7802170,
    'kidneys': 356788,
    'spleen': 271253,
    'psoas_majors': 442497,
    'thyroid_gland': 22580,
    'pancreas': 85428
}


class VisceralDataset(Dataset):
    def __init__(self, dataset_path, valid_seg_path, patient_id, organs, transforms=None, resample_voxels=True):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.organs = organs
        self.organ_ids = [ORGANS[organ]['id'] for organ in self.organs]
        self.patient_id = patient_id
        self.transforms = transforms
        self.resample_voxels = resample_voxels
        
        with open(valid_seg_path) as f:
            valid_segs = json.load(f)
            self.valid_idxs = set()
            for organ_id in self.organ_ids:
                self.valid_idxs |= set(valid_segs[PATIENT_LIST[patient_id]][str(organ_id)])
            self.valid_idxs = list(self.valid_idxs)

        self.data_cache = [None] * len(self.valid_idxs)
        print("Dataset for user <{}> organs <{}> initiated".format(patient_id, organs))

    def __getitem__(self, idx):
        if self.data_cache[idx]:
            image_np, mask_np = self.data_cache[idx]
        else:
            with h5py.File(self.dataset_path, 'r') as dataset:
                volume = dataset['volumes'][PATIENT_LIST[self.patient_id]]
                segmentation = dataset['segmentations'][PATIENT_LIST[self.patient_id]]
                
                image_file = Image.fromarray(np.array(volume[:, self.valid_idxs[idx], :], dtype=np.float32))
                mask_file = Image.fromarray(np.array(segmentation[:, self.valid_idxs[idx], :], dtype=np.float))
                
                
                if self.transforms is not None:
                    image_file = self.transforms(image_file)
                    mask_file = self.transforms(mask_file)

                image_np = np.array(image_file)
                mask_np = self._fix_classes(np.array(mask_file, dtype=np.int)) # Hide other classes
                
                image_np = self._preprocess_ct_img(image_np)
                mask_np = self._preprocess_mask_img(mask_np)
                
                # image_np = tr_F.to_tensor(image_np).float()
                # image_np = tr_F.normalize(image_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                image_np = tr_F.to_tensor(image_np).float()
                mask_np = torch.tensor(mask_np).long()
            self.data_cache[idx] = (image_np, mask_np)

        return image_np, mask_np
    
    def __len__(self):
        return len(self.valid_idxs)
    
    def _preprocess_ct_img(self, img_slc):
        """
        Set pixels with hounsfield value great than 1200, to zero.
        Clip all hounsfield values to the range [-100, 400]
        Normalize values to [0, 1]
        Rescale img and label slices to 388x388
        Pad img slices with 92 pixels on all sides (so total shape is 572x572)
        
        Args:
            img_slc: raw image slice
        Return:
            Preprocessed image slice
        """      
        img_slc = img_slc.astype(np.float)
        img_slc = normalize_image(img_slc)

        return img_slc
    
    def _preprocess_mask_img(self, msk):
        """ Preprocess ground truth slice to match output prediction of the network in terms 
        of size and orientation.
        
        Args:
            lbl_slc: raw label/ground-truth slice
        Return:
            Preprocessed label slice"""
        msk = msk.astype(np.uint8)
        return msk
    
    def _fix_classes(self, mask):
        mask = mask * np.isin(mask, [0] + self.organ_ids) # Remove other classes
        
        # reform to the 0 - n interval
        mask = mask + 100
        for i,organ_id in enumerate([0] + self.organ_ids):
            mask[mask == organ_id + 100] = i
        return mask


class VisceralVolumeDataset(Dataset):
    def __init__(self, is_silver):
        super().__init__()
        if is_silver:
            self.dataset_path = cfg['visceral']['silver_path']
            self.patient_ids = list(map(str, range(len(SILVER_PATIENT_LIST))))
        else:
            self.dataset_path = cfg['visceral']['path']
            self.patient_ids = list(map(str, range(len(PATIENT_LIST))))
        
        with h5py.File(self.dataset_path, 'r') as dataset:
            self.volumes = []
            for patient_id in self.patient_ids:
                vol = dataset['volumes'][patient_id][:]
                vol_mean = np.mean(vol)
                vol_std = np.std(vol)
                vol = (vol - vol_mean) / vol_std
                self.volumes.append(vol)
        
    def __getitem__(self, patient_idx):
        return self.volumes[patient_idx]

    def __len__(self):
        return len(self.volumes)

def get_visceral_medical_few_shot_dataset(organs, patient_ids, shots, transforms):
    viscerals = []
    for organ in organs:
        datasets = []
        for patient_id in patient_ids:
            dataset = VisceralDataset(dataset_path=cfg['visceral']['path'],valid_seg_path=cfg['visceral']['valid_seg'],patient_id=patient_id, organs=[organ], transforms=transforms)
            if len(dataset):
                datasets.append(dataset)
        viscerals.append(datasets)
    dataset = MedicalFewshotDataset(viscerals, shots=shots)
    return dataset

def get_normal_medical_dataset(organs, patient_ids, transforms):
    datasets = []
    for patient_id in patient_ids:
        dataset = VisceralDataset(dataset_path=cfg['visceral']['path'],valid_seg_path=cfg['visceral']['valid_seg'],patient_id=patient_id, organs=organs, transforms=transforms)
        if len(dataset):
            datasets.append(dataset)
    dataset = MedicalNormalDataset(datasets)
    return dataset
