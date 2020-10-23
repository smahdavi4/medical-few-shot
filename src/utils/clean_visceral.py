import os
import re
import sys
import subprocess
import h5py
import nibabel as nib
import numpy as np
import json
import ants

SEG_REGEX = '{patient_id}_{organ_ids}_.*gz'
IS_SILVER = False

silver_patient_list = [
    '10000101_1_CTce_ThAb',
    # '10000107_1_CTce_ThAb',  # No Segmentations
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
    # '10000157_1_CTce_ThAb',  # No segmentation
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

gold_patient_list = [
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
    '10000129_1_CTce_ThAb',  # Broken
    '10000132_1_CTce_ThAb',
    '10000112_1_CTce_ThAb',
    '10000111_1_CTce_ThAb',
    '10000131_1_CTce_ThAb'
]

organs = {  # RadlexID
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

hf = h5py.File('visceral.h5', 'w')
vol_group = hf.create_group('volumes')
seg_group = hf.create_group('segmentations')


def save_h5(patient_id, vol, seg):
    vol_group.create_dataset(patient_id, data=vol, compression="gzip")
    seg_group.create_dataset(patient_id, data=seg, compression="gzip")


def crop_center(img, cropx, cropy, cropz):
    print(img.shape)
    x, y, z = img.shape
    start_x = x // 2 - (cropx // 2)
    start_y = y // 2 - (cropy // 2)
    start_z = z // 2 - (cropz // 2)
    return img[start_y:start_y + cropy, start_x:start_x + cropx, start_z:start_z + cropz]


def read_nib(addr):
    # img = np.array(nib.load(addr).dataobj)
    img = nib.load(addr)
    ants_data = ants.from_nibabel(img)
    resampled = ants.resample_image(ants_data, (1.5, 1.5, 1.5)).numpy()  # Resample to 1.5mm3 resolution
    print("Resampled Size: ", resampled.shape)
    np_resampled_pad = np.pad(resampled, 100, 'constant', constant_values=resampled.min())
    # np_resampled_pad = np.pad(img, 100, 'constant', constant_values=img.min())
    return crop_center(np_resampled_pad, 256, 256, 400)


def create_new_dataset(vol_dir, seg_dir, patient_list):
    vol_list = os.listdir(vol_dir)
    seg_list = os.listdir(seg_dir)

    for patient_id in patient_list:

        # Volume
        patient_vol = read_nib('{}/{}.nii.gz'.format(vol_dir, patient_id))

        # Seg
        patient_seg = np.zeros(patient_vol.shape, dtype=np.int8)

        for organ in reversed(list(organs.keys())):
            organ_vol = np.zeros(patient_seg.shape, dtype=np.int8)
            if IS_SILVER:
                organ_regex = re.compile(SEG_REGEX.format(patient_id=patient_id[:-10],
                                                          organ_ids='(' + '|'.join(organs[organ]['radlex']) + ')'))
            else:
                organ_regex = re.compile(
                    SEG_REGEX.format(patient_id=patient_id, organ_ids='(' + '|'.join(organs[organ]['radlex']) + ')'))
            organ_files = list(filter(organ_regex.match, seg_list))

            if not organ_files:
                print("No organ segmentation for patient: {}, organ: {}".format(patient_id, organ))
            else:
                print("Organ files for patient: {}, organ: {}, files: {}".format(patient_id, organ, organ_files))
            for file in organ_files:
                file_path = os.path.join(seg_dir, file)
                organ_seg_vol = read_nib(file_path)
                organ_vol[np.array(organ_seg_vol, dtype=np.bool)] = 1

            patient_seg[organ_vol == 1] = organs[organ]['id']

        save_h5(patient_id, patient_vol, patient_seg)
        print("Patient {} Done".format(patient_id))


def dl_gold_dataset(user, passw):
    subprocess.call(
        'wget -r -A "*CTce*" ftp://{}:{}@153.109.124.90/visceral-dataset/Anatomy3-trainingset/Volumes'.format(user,
                                                                                                              passw),
        shell=True
    )
    subprocess.call(
        'wget -r -A "*CTce*" ftp://{}:{}@153.109.124.90/visceral-dataset/Anatomy3-trainingset/Segmentations'.format(
            user, passw),
        shell=True
    )


def dl_silver_dataset(user, passw):
    subprocess.call(
        'wget -r -A "*CTce*" ftp://{}:{}@153.109.124.90/visceral-dataset/SilverCorpus/Volumes'.format(user, passw),
        shell=True
    )
    subprocess.call(
        'wget -r -A "100001*,100002*" ftp://{}:{}@153.109.124.90/visceral-dataset/SilverCorpus/Segmentations'.format(
            user, passw),
        shell=True
    )


def save_non_empty_indices(patient_list):
    hf_read = h5py.File('visceral.h5', 'r')
    all_seg = hf_read['segmentations']
    valid_organs = {}
    for patient_id in patient_list:
        pat_seg = all_seg[patient_id]
        valid_organs[patient_id] = {}
        for i in range(pat_seg.shape[1]):
            present_organs = np.unique(pat_seg[:, i, :])
            for present_organ in present_organs:
                if present_organ == 0:  # It is a background
                    continue
                valid_organs[str(patient_id)][str(present_organ)] = valid_organs[str(patient_id)].get(
                    str(present_organ), [])
                valid_organs[str(patient_id)][str(present_organ)].append(i)
        print("Valid organs for patient {} done.".format(patient_id))
    with open('valid_seg.json', 'w') as f:
        json.dump(valid_organs, f)


if __name__ == "__main__":
    user = sys.argv[1]
    passw = sys.argv[2]

    visceral_folder = '153.109.124.90'

    if not IS_SILVER:  # Gold corpus
        vol_dir = '{}/visceral-dataset/Anatomy3-trainingset/Volumes'.format(visceral_folder)
        seg_dir = '{}/visceral-dataset/Anatomy3-trainingset/Segmentations'.format(visceral_folder)
        # dl_gold_dataset(user, passw)
        create_new_dataset(vol_dir, seg_dir, gold_patient_list)

    else:  # Silver corpus
        vol_dir = '{}/visceral-dataset/SilverCorpus/Volumes'.format(visceral_folder)
        seg_dir = '{}/visceral-dataset/SilverCorpus/Segmentations'.format(visceral_folder)
        # dl_silver_dataset(user, passw)
        create_new_dataset(vol_dir, seg_dir, silver_patient_list)

    hf.close()

    # if not IS_SILVER:
    #     save_non_empty_indices(gold_patient_list)
    # else:
    #     save_non_empty_indices(silver_patient_list)
