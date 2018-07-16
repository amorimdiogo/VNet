"""
Diogo Amorim, 2018-07-13
Utilities
"""

import h5py
import matplotlib.pyplot as plt


def print_slice(data, truth, slice):
    fig = plt.figure()
    y = fig.add_subplot(1, 2, 1)
    y.imshow(data[:, :, slice, 0], cmap='gray')
    y.set_title('Volume')
    y.set_xlabel('x axis')
    y.set_ylabel('y axis')
    y = fig.add_subplot(1, 2, 2)
    # y.imshow(truth[:, :, slice, 0]*data[:, :, slice, 0], cmap='gray')
    y.imshow(truth[:, :, slice, 0], cmap='gray')
    y.set_title('Segmentation')
    y.set_xlabel('x axis')
    y.set_ylabel('y axis')
    plt.show


def load_dataset(path, h5=True):
    #print("Loading dataset... Shape:")
    file = h5py.File(path, 'r')
    if h5:
        data = file.get('data')
        truth = file.get('truth')
    else:
        data = file.get('data').value
        truth = file.get('truth').value
    # print(data.shape)
    return data, truth
