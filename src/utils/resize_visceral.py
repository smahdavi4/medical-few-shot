import sys
import numpy as np
import h5py
import cv2

image_shape = (256, 256)

if __name__ == "__main__":
    visceral_path = sys.argv[1]
    visceral_new_path = sys.argv[2]
    
    new_hf = h5py.File(visceral_new_path, 'w')
    vol_group = new_hf.create_group('volumes')
    seg_group = new_hf.create_group('segmentations')
    
    old_hf = h5py.File(visceral_path, 'r')
    for patient_id in list(old_hf['volumes'].keys()):
        if patient_id == '10000129_1_CTce_ThAb': # Broken (Gold)
            continue
        if patient_id in ['10000107_1_CTce_ThAb', '10000157_1_CTce_ThAb']: # Broken (Silver)
            continue
        vol = old_hf['volumes'][patient_id][:]
        seg = old_hf['segmentations'][patient_id][:]
        
        resized_vol = np.array([cv2.resize(vol[:, :, i], image_shape) for i in range(vol.shape[2])], dtype=np.int32)
        resized_seg = np.array([cv2.resize(seg[:, :, i], image_shape, interpolation=cv2.INTER_NEAREST) for i in range(vol.shape[2])], dtype=np.int8)
        
        vol_group.create_dataset(patient_id, data=resized_vol, compression="gzip")
        seg_group.create_dataset(patient_id, data=resized_seg, compression="gzip")
        print("Patient ", patient_id, " Done")
    
    new_hf.close()
    old_hf.close()
