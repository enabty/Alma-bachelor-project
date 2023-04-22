from astropy.io import fits
import numpy as np
import glob
from helper_functions import *


"""

This function takes a path to a folder containing FITS files and returns the data in 
an array of 2D-arrays of size 000x100. Ready to be used as input for a neural network.

Input:

    file_path: path to folder containing FITS files
    linnear_aug: if True, the data will be augmented by linear transformations
    augment_factor: how many times the data will be augmented

"""


def init_training_data_from_folder(file_path, linnear_aug=False, augment_factor=5):

    fits_files_data = [fits.getdata(file).squeeze() for file in glob.glob(file_path + '/*.fits')]
    fits_files_data = [crop_around_middle_50x50_percent(file) for file in fits_files_data if file.shape[0] > 500 and file.shape[1] > 500]
    fits_files_data = ([crop_around_max_value_400x400(file) for file in fits_files_data])
    if linnear_aug:
        return np.array([linear_transformation(file) for file in fits_files_data for i in range(augment_factor) if file.shape == (400, 400)])
    return np.array([crop_middle_100x100(file) for file in fits_files_data if file.shape == (400, 400)])


"""

Convert a folder of FITS files to a numpy array and save it to a npy file.

"""

def fits_to_npy(folder_path, npy_path, linnear_aug=False, augment_factor=5): np.save(npy_path, init_training_data_from_folder(folder_path, linnear_aug, augment_factor), allow_pickle=True)

"""

Init X and y from numpy arrays of positive and negative data. 

"""


def init_training_data_from_npy(pos_path, neg_path, linnear_aug=False, augment_factor=5):

    fits_pos = np.load(pos_path, allow_pickle=True)
    fits_neg = np.load(neg_path, allow_pickle=True)

    if linnear_aug:
        fits_pos = np.array([linear_transformation(file) for file in fits_pos for i in range(augment_factor) if file.shape == (400, 400)])
        fits_neg = np.array([linear_transformation(file) for file in fits_neg for i in range(augment_factor) if file.shape == (400, 400)])

    y = [0] * len(fits_neg) + [1] * len(fits_pos)

    X = np.concatenate((fits_neg, fits_pos), axis=0)

    return X, y

 

__name__ == '__main__' and print('pre_processing.py is working')