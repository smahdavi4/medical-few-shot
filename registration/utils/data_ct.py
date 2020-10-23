import h5py
import numpy as np
import PIL


def read_png(path):  # reads a single png image as numpy array
    return np.asarray(PIL.Image.open(path))


def read_X_y_list(hf, patients, image_shape):
    X_data = []
    y_data = []

    for p in patients:
        pat_vol = np.array(hf['volumes'][p][:], dtype=np.float32).swapaxes(1, 2).swapaxes(0, 1)
        pat_seg = np.array(hf['segmentations'][p][:], dtype=np.int8).swapaxes(1, 2).swapaxes(0, 1)

        normalized_vol = (pat_vol - pat_vol.min()) / (pat_vol.max() - pat_vol.min())  # normalize the whole 3d image

        X_data.append(normalized_vol)
        y_data.append(pat_seg)

    return X_data, y_data


def read_data(visceral_path, image_shape, split):
    # chaos_path: the path to the inside of the chaos directory
    # image_shape: shape of each slice
    # split: [train_count, valid_count, test_count]

    # returns (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

    hf_data = h5py.File(visceral_path, 'r')
    patients = list(hf_data['volumes'].keys())

    # `split` should have an element for each of the 'train', 'validation' and 'test' summing to `len(patients)`
    assert len(split) == 3
    assert sum(split) <= len(patients)

    train_count = split[0]
    valid_count = split[1]
    test_count = split[2]

    # split
    train_patients = patients[:train_count]
    valid_patients = patients[train_count:train_count + valid_count]
    if test_count:
        test_patients = patients[-test_count:]
    else:
        test_patients = []

    # read data from files
    train_data = read_X_y_list(hf_data, train_patients, image_shape)
    valid_data = read_X_y_list(hf_data, valid_patients, image_shape)

    test_data = read_X_y_list(hf_data, test_patients, image_shape)

    hf_data.close()

    # return all
    return train_data, valid_data, test_data
