"""
Diogo Amorim, 2018-07-13
Pré-processamento do Dataset LiTS para tumores no fígado - Vnet

- Converts .nii file to a resized ndarray
- HU augmentation
- Resize layers
- Normalization [0, 1]
- ...
"""

import os
import re

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.misc
import scipy.ndimage

INPUT_SIZE = 256  # Input feature width/height
INPUT_DEPTH = 64  # Input depth

""" K. Sahi, S. Jackson, E. Wiebe, G. Armstrong, S. Winters, R. Moore, et al., ”The value of liver windows
settings in the detection of small renal cell carcinomas on unenhanced computed tomography,”
Canadian Association of Radiologists Journal, vol. 65, pp. 71-76, 2014."""
MIN_HU = -160  # Min HU Value
MAX_HU = 240  # Max HU Value


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def hu_window(image):
    image[image < MIN_HU] = MIN_HU
    image[image > MAX_HU] = MAX_HU
    image = (image - MIN_HU) / (MAX_HU - MIN_HU)
    image = image.astype("float32")
    return image


def scale_volume(volume, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE, hu_value=True):
    if hu_value:
        volume = hu_window(volume)

    size_scale_factor = img_px_size / volume.shape[0]
    depth_scale_factor = img_depth / volume.shape[-1]

    volume = scipy.ndimage.interpolation.rotate(volume, 90, reshape=False)

    vol_zoom = scipy.ndimage.interpolation.zoom(volume, [size_scale_factor, size_scale_factor,
                                                         depth_scale_factor], order=1)

    vol_zoom[vol_zoom < 0] = 0
    vol_zoom[vol_zoom > 1] = 1
    return vol_zoom


def scale_segmentation(segmentation, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE):
    size_scale_factor = img_px_size / segmentation.shape[0]
    depth_scale_factor = img_depth / segmentation.shape[-1]

    segmentation = scipy.ndimage.interpolation.rotate(segmentation, 90, reshape=False)

    # Nearest neighbour is used to mantain classes discrete
    seg_zoom = scipy.ndimage.interpolation.zoom(segmentation, [size_scale_factor, size_scale_factor,
                                                               depth_scale_factor], order=0)

    return seg_zoom


def get_data(vol_dir, seg_dir, crop=False):
    volume = nib.load(vol_dir).get_data()
    segmentation = nib.load(seg_dir).get_data()

    if crop:
        aux = []
        for i in range(segmentation.shape[2]):
            if np.sum(segmentation[:, :, i]) > 0:
                aux.append(i)

        volume = volume[:, :, (np.min(aux)-1):(np.max(aux)+1)]
        segmentation = segmentation[:, :, (np.min(aux)-1):(np.max(aux)+1)]

    return volume, segmentation


def create_dataset(path, px_size=INPUT_SIZE, slice_count=INPUT_DEPTH, crop=False, normalize=False):
    """Returns dataset with shape (m, z, x, y, n)"""

    files = sorted_alphanumeric(os.listdir(path))
    segmentations = []
    volumes = []
    for name in files:
        if name[0] == 's':
            segmentations.append(name)
        elif name[0] == 'v':
            volumes.append(name)

    # m = len(volumes)
    m = 8
    if crop:
        print("Creating Cropped Data Set:")
    else:
        print("Creating Data Set:")

    slices = []
    print("0/%i (0%%)" % m)
    for i in range(m):
        volume, segmentation = get_data(path + volumes[i], path + segmentations[i], crop)

        volume = scale_volume(volume, slice_count, px_size)
        print(volume.shape)
        segmentation = scale_segmentation(segmentation, slice_count, px_size)
        slices.append([volume, segmentation])
        print("%i/%i (%i%%)" % (i+1, m, ((i+1)/m*100)))

    dataset = np.array(slices)

    print("Dataset finished with shape:")
    print(dataset.shape)

    return dataset


def write_dataset(data_set, path):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset('data', data=np.expand_dims(data_set[:, 0, :, :, :], -1))
    truth = np.expand_dims(data_set[:, 1, :, :, :], -1)
    truth = divide_segmentation(truth)
    h5f.create_dataset('truth', data=truth)
    h5f.close()
    print("Dataset saved @ %s" % path)


def divide_segmentation(segmentation):
    layer1 = np.copy(segmentation)
    layer2 = np.copy(segmentation)
    background = np.copy(segmentation)

    layer1[layer1==2] = 1
    layer2[layer2==1] = 0
    layer2[layer2 > 0] = 1
    background[background==1] = 5
    background[background == 0] = 1
    background[background>1] = 0
    # np.concatenate((layer1, layer2, background), axis=-1)
    # return layer1, layer2, background
    a = np.concatenate((layer1, layer2, background), axis=-1)
    # return a
    return layer1  # liver


data_dir = "/home/lits_dataset/train_batch/"
train_dir = data_dir + "train/"
val_dir = data_dir + "val/"
# test_dir = "/media/albaroz/diogo-pen/Training_Batch_1/"
save_dir = "/home/guest/PycharmProjects/tese/Vnet/dataset/"

# print("Obtaining Training Data:")
# train_set = create_dataset(train_dir, crop=False)
# write_dataset(train_set, save_dir + "train_data.h5")

print("Obtaining Validation Data:")
val_set = create_dataset(val_dir)
write_dataset(val_set, save_dir + "val_data.h5")

# print("Obtaining Test Data:")
# test_set = create_dataset(test_dir)
# write_dataset(test_set, save_dir + "test_data.h5")
