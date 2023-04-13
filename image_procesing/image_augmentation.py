import numpy as np
import random

from scipy.ndimage import rotate
from skimage.transform import resize


def crop_around_max_value_400x400(fits):
    (x, y) = np.unravel_index(np.argmax(fits, axis=None), fits.shape)
    return fits[x-200:x+200, y-200:y+200]


def geometric_mean_square(a, b): return np.sqrt(np.multiply(abs(a), abs(b)))

# _------ It is ugly atm, but it works ATM. Reowrk so it is shorter ------------_#
# Linnear transformation || Random  rezise, rotate, flip and return 100x100


def linear_transformation(fits):
    ret_image = fits

    random_resize = random.randint(150, 350)
    ret_image = resize(ret_image, (random_resize, random_resize))

    ret_image = rotate(ret_image, random.randint(0, 360), reshape=False)

    if random.getrandbits(1):
        ret_image = np.fliplr(ret_image)
    if random.getrandbits(1):
        ret_image = np.flipud(ret_image)

    (lwr_bound, upr_bound) = int(random_resize/2) - 55, int(random_resize/2) - 45
    x = random.randint(lwr_bound, upr_bound)
    y = random.randint(lwr_bound, upr_bound)

    return ret_image[x:x+100, y:y+100]


def pos_image_augmentation(pos_data):

    # resize the data so that all the disks are roughly the same size exepct for the two largest ones
    pos_data = [file if (file is pos_data[2] or file is pos_data[3]) else resize(
        file, (2000, 2000)) for file in pos_data]

    # Rotate the data so that the stars are aligned according to predefined angles
    pos_data = [rotate(data, degrees, reshape=False) for (data, degrees) in list(
        zip(pos_data, [0, 0, -15, -15, -9, -9, -13, -13]))]

    # This now we have a list of 100x100 matrices where the star is in the middle and aligned
    pos_data = [crop_around_max_value_400x400(file) for file in pos_data]

    # Flip each image LR, UD and both
    f1, f2, f3, f4 = lambda file: file, lambda file: np.fliplr(
        file), lambda file: np.flipud(file), lambda file: np.flip(file)
    # flip_fits = lambda file: [f(file) for f in [f1, f2, f3, f4]]
    pos_data = [f(file) for file in pos_data for f in [f1, f2, f3, f4]]

    # Multiply two matrices together and return the square root of the result
    # Make non-linnear combinations of all the fits files
    pos_data += [geometric_mean_square(pos_data[i], pos_data[j])
                 for i in range(0, len(pos_data)) for j in range(i+4, len(pos_data))]

    pos_data += [linear_transformation(file)
                 for file in pos_data for i in range(0, 5)]

    return pos_data


def neg_image_augmentation(neg_data):
    neg_data = [crop_around_max_value_400x400(file) for file in neg_data]
    neg_data += [linear_transformation(file)for file in neg_data for i in range(
        0, 100) if file.shape == (400, 400)]
    return neg_data


def image_augmentation(data):
    if len(data) < 20:
        return pos_image_augmentation(data)
    return neg_image_augmentation(data)


"""

Improvements:

In image we should let the if statement be based on the length of pos_data instead of the length of data.

"""

__name__ == '__main__' and print('pre_processing.py is working')
