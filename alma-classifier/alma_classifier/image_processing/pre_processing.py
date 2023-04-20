from astropy.io import fits
import numpy as np
import glob
from image_processing.image_augmentation import linear_transformation


def crop_around_middle_50x50_percent(fits): return fits[(int(fits.shape[0]*.25)):(int(fits.shape[0]*.75)),
                                                        (int(fits.shape[1]*.25)):(int(fits.shape[1]*.75))]

def crop_middle_100x100(fits): return fits[fits.shape[0]//2-50:fits.shape[0]//2+50,
                                            fits.shape[1]//2-50:fits.shape[1]//2+50]


def crop_around_max_value_400x400(fits):
    (x, y) = np.unravel_index(np.argmax(fits, axis=None), fits.shape)
    return fits[x-200:x+200, y-200:y+200]




"""

This function takes a path to a folder containing FITS files and returns the data in 
an array of 2D-arrays of size 400x400.


"""


def init_fits_files_from_folder(file_path):
    fits_files_data = [fits.getdata(file).squeeze() for file in glob.glob(file_path + '/*.fits')]
    print(len(fits_files_data))
    fits_files_data = [crop_around_middle_50x50_percent(file) for file in fits_files_data if file.shape[0] > 500 and file.shape[1] > 500]
    return np.array([crop_around_max_value_400x400(file) for file in fits_files_data])



"""

Init X and y from numpy arrays of positive and negative data. 

"""


def init_from_npy(pos_path, neg_path):

    fits_pos = np.load(pos_path, allow_pickle=True)
    fits_pos = [fits for fits in fits_pos if fits.shape == (400, 400)]
    fits_pos = np.array(fits_pos)

    fits_neg = np.load(neg_path, allow_pickle=True)
    fits_neg = [fits for fits in fits_neg if fits.shape == (400, 400)]
    fits_neg = np.array(fits_neg)

    y = [0] * len(fits_neg) + [1] * len(fits_pos)

    X = np.concatenate((fits_neg, fits_pos), axis=0)

    X = [linear_transformation(fits) for fits in X]

    return X, y

 
def gen_neg_dataset():
    data = init_fits_files_from_folder(
        'C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg')
    print(len(data))
    np.save('C:/ChalmersWorkspaces/KandidatArbete/Alma-bachelor-project/data/neg_dataset/neg_dataset.npy', data, allow_pickle=True)

__name__ == '__main__' and print('pre_processing.py is working')