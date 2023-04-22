
import numpy as np
import random
from scipy.ndimage import rotate
from skimage.transform import resize

"""

Crop functions for the fits files. Self explanatory.

"""

def crop_around_middle_50x50_percent(fits): return fits[(int(fits.shape[0]*.25)):(int(fits.shape[0]*.75)),
                                                        (int(fits.shape[1]*.25)):(int(fits.shape[1]*.75))]


def crop_middle_100x100(fits): return fits[fits.shape[0]//2-50:fits.shape[0]//2+50,
                                           fits.shape[1]//2-50:fits.shape[1]//2+50]


def crop_around_max_value_400x400(fits):
    (x, y) = np.unravel_index(np.argmax(fits, axis=None), fits.shape)
    return fits[x-200:x+200, y-200:y+200]


"""

Geometric mean square function. Used to make non-linnear combinations of the fits files. To generate more data.

"""


def geometric_mean_square(a, b): return np.sqrt(np.multiply(abs(a), abs(b)))



"""

Linnear transformation || Random  rezise, rotate, flip and return 100x100

"""


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


__name__ == '__main__' and print('helper_functions.py is working')



