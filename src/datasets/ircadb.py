import os
import logging
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tr_F

import pydicom

from .base import FewShotDataset, MedicalFewshotDataset
from utils.image import to_scale, normalize_image
from config import cfg

ORGAN_TYPES = ["artery", "liver", "livertumor03", "livertumor07", "skin", "bone", "liverkyst", "livertumor04", "portalvein", "spleen", "leftkidney", "livertumor01", "livertumor05", "rightkidney", "venoussystem", "leftlung", "livertumor02", "livertumor06", "rightlung"]

ORGANS = ["liver", "bone", "spleen", "leftkidney", "rightkidney", "leftlung", "rightlung"]


class IrcadbDataset(Dataset):
    def __init__(self, root, patient_ids, organ, transforms=None):
        self.root = root
        self.organ = organ
        self.patient_ids = patient_ids
        self.transforms = transforms
        
        self.img_file_list, self.mask_file_list = self.get_img_mask_path(organ=organ, patient_id_list=patient_ids, shuffle=False)
        
        assert len(self.img_file_list) == len(self.mask_file_list)
        
    def __getitem__(self, idx):
        image_file = Image.fromarray(pydicom.read_file(self.img_file_list[idx]).pixel_array)
        mask_file = Image.fromarray(pydicom.read_file(self.mask_file_list[idx]).pixel_array)
        
        if self.transforms is not None:
            image_file = self.transforms(image_file)
            mask_file = self.transforms(mask_file)

        image_np = np.array(image_file)
        mask_np = np.array(mask_file) > 0
        
        image_np = self._preprocess_ct_img(image_np)
        mask_np = self._preprocess_mask_img(mask_np)
        
        # image_np = tr_F.to_tensor(image_np).float()
        # image_np = tr_F.normalize(image_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        image_np = tr_F.to_tensor(image_np).float()
        mask_np = torch.tensor(mask_np).long()
        
        return image_np, mask_np
    
    def __len__(self):
        return len(self.img_file_list)
    
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
        # The following preprocessings Only applies for a specific organ
        # img_slc[img_slc>1200] = 0 
        # img_slc = np.clip(img_slc, -100, 400)
        img_slc = normalize_image(img_slc)
        # img_slc = to_scale(img_slc, (400, 400))
        # img_slc = np.pad(img_slc,((92,92),(92,92)),mode='reflect')

        return img_slc
    
    def _preprocess_mask_img(self, msk):
        """ Preprocess ground truth slice to match output prediction of the network in terms 
        of size and orientation.
        
        Args:
            lbl_slc: raw label/ground-truth slice
        Return:
            Preprocessed label slice"""
        msk = msk.astype(np.uint8)
        #scale the label slc for comparison with the prediction
        # msk = to_scale(msk , (400, 400))
        return msk
    
    def get_img_mask_path(self, organ, patient_id_list=None, shuffle=True):
        '''
        returns a list of slices and their corresponding masks for the given organ.
        skips the slices having an empty mask (organ is not present in those slices)
        '''
        slice_path_list = []
        mask_path_list = []

        for patient_id in patient_id_list:
            patient_dicom_path = "3Dircadb1." + str(patient_id) + "/PATIENT_DICOM"
            mask_dicom_path = "3Dircadb1." + str(patient_id) + "/MASKS_DICOM/" + organ
            slice_path = os.path.join(self.root, patient_dicom_path)
            mask_path = os.path.join(self.root, mask_dicom_path)
            
            slices = []
            masks = []
            empty_masks = set()
            
            try:
                for mask in sorted(os.listdir(mask_path), key=lambda x: int(x[6:])):
                    single_mask_path = os.path.join(mask_path, mask)
                    if os.path.basename(single_mask_path)[0:5] == "image":
                        mask_pixels = pydicom.read_file(single_mask_path).pixel_array
                        if not np.any(mask_pixels):
                            empty_masks.add(single_mask_path)
                        masks.append(single_mask_path)
            except FileNotFoundError:
                logging.warning("patient id {} has no organ named '{}'".format(patient_id, organ))
                continue
            
            for slice in sorted(os.listdir(mask_path), key=lambda x: int(x[6:])):
                single_slice_path = os.path.join(slice_path, slice)
                if os.path.basename(single_slice_path)[0:5] == "image":
                    
                    slices.append(single_slice_path)

            for slice, mask in zip(slices, masks):
                if mask not in empty_masks:
                    slice_path_list.append(slice)
                    mask_path_list.append(mask)
            
            if shuffle == True:
                c = list(zip(slice_path_list, mask_path_list))
                random.shuffle(c)
                slice_path_list, mask_path_list = zip(*c)
                slice_path_list = list(slice_path_list)
                mask_path_list = list(mask_path_list)

        return slice_path_list, mask_path_list

def get_ircadb_few_shot_datasets(organs, patient_ids, iterations, N_shot, N_query, transforms):
    irdcadbs = [IrcadbDataset(root=cfg['ircadb']['root'],patient_ids=patient_ids, organ=organ, transforms=transforms) for organ in organs]
    dataset = FewShotDataset(irdcadbs, iterations, N_shot, N_query)
    return dataset

def get_ircadb_medical_few_shot_dataset(organs, patient_ids, shots, mode, transforms):
    irdcadbs = []
    for organ in organs:
        datasets = []
        for patient_id in patient_ids:
            dataset = IrcadbDataset(root=cfg['ircadb']['root'],patient_ids=[patient_id], organ=organ, transforms=transforms)
            if len(dataset):
                datasets.append(dataset)
        irdcadbs.append(datasets)
    dataset = MedicalFewshotDataset(irdcadbs, mode=mode, shots=shots)
    return dataset
