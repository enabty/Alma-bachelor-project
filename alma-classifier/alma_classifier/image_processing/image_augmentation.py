import numpy as np
import glob
import astropy.io.fits as fits
from scipy.ndimage import rotate
from skimage.transform import resize
from .support_functions import *



def generate_unsorted_dataset(folder_path):

    pos_data = [fits.getdata(file).squeeze() for file in glob.glob(folder_path + '/*.fits')]

    # resize the data so that all the disks are roughly the same size exepct for the two largest ones
    pos_data = [file if (file is pos_data[2] or file is pos_data[3]) else resize(
        file, (2000, 2000)) for file in pos_data]

    # Rotate the data so that the stars are aligned according to predefined angles
    pos_data = [rotate(data, degrees, reshape=False) for (data, degrees) in list(
        zip(pos_data, [0, 0, -15, -15, -9, -9, -13, -13]))]

    pos_data = [crop_around_max_value_400x400(file) for file in pos_data]

    # Flip each image LR, UD and both
    f1, f2, f3, f4 = lambda file: file, lambda file: np.fliplr(
        file), lambda file: np.flipud(file), lambda file: np.flip(file)
    pos_data = [f(file) for file in pos_data for f in [f1, f2, f3, f4]]

    # Make non-linnear combinations of all the fits files
    pos_data += [geometric_mean_square(pos_data[i], pos_data[j])
                 for i in range(0, len(pos_data)) for j in range(i+1, len(pos_data))]

    return pos_data


__name__ == '__main__' and print('pre_processing.py is working')
