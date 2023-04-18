
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from clustar.core import ClustarData
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from scipy.ndimage import rotate
from PIL import Image
from scipy import interpolate
from operator import itemgetter



def change_contrast(image, multiplicand = 1, addend = 0):
    if (image.file_format != ".fits"):
        print("file must be .fits format")
        return
    image.img_data = (image.img_data * multiplicand) + addend

    return image.img_data

def noise_jitter(image, percetage_changed=1.0):
    if (image.file_format != ".fits"):
        print("file must be .fits format")
        return
    
    data = image.img_data[0][0]
    indices = np.asarray([(i,j) for i in range(len(data)) for j in range(len(data[i]))])
    np.random.shuffle(indices)
    n = int(np.floor(data.shape[0]*data.shape[1]*percetage_changed))
    for i in range(n):
        x = indices[i][0]
        y = indices[i][1]
        data[x][y] *= random.uniform(0.97,1.03)

    return image.img_data

def neighbour_jitter(image, percetage_changed=1.0):
    data = image.img_data[0][0]
    indices = np.asarray([(i,j) for i in range(len(data)) for j in range(len(data[i]))])
    np.random.shuffle(indices)
    n = int(np.floor(data.shape[0]*data.shape[1]*percetage_changed))

    change_buffer = []
    for i in range(n):
        y = indices[i][0]
        x = indices[i][1]
        neighbours = np.asarray([data[max(0, y-1)][x], data[min(data.shape[0]-1, y+1)][x], data[y][max(0, x-1)], data[y][min(data.shape[0]-1, x+1)]])
        max_intensity = np.max(neighbours)
        min_intensity = np.min(neighbours)
        change_buffer.append((y,x, random.uniform(min_intensity,max_intensity)))
    
    for y, x, intensity in change_buffer:
        data[y][x] = intensity

    return image.img_data

    

# Cut out a square of a radius at a center = (x,y) and scale up the size with interpolation
def crop_resize(image, radius, center):
    if (image.file_format != ".fits"):
        print("file must be .fits format")
        return

    matrix = image.img_data[0][0]
    mini = np.nanmin(matrix)
    maxi = np.nanmax(matrix)
    # Radius cannot be bigger than the closest edge
    radius = min(radius, center[0], image.img_data.shape[2] - center[0], center[1], image.img_data.shape[3] - center[1])
    # Cut the matrix in the center
    focus = matrix[center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius]
    focus = np.nan_to_num(focus)
    # Interpolate up to our 100x100 standard size
    x = np.linspace(mini,maxi,radius*2)
    y = np.linspace(mini,maxi,radius*2)
    f = interpolate.interp2d(x,y,focus,kind='cubic')
    x2 = np.linspace(mini, maxi, 100)
    y2 = np.linspace(mini, maxi, 100)
    arr2 = f(y2, x2)
    arr2.shape = (1,1,100,100)
    #image.img_data = arr2
    return arr2

def standard_crop(image, center):
    crop_size = units.Quantity((100, 100), units.pixel)
    img_crop = Cutout2D(image.img_data[0][0], center, crop_size).data
    img_crop.shape = (1,1,img_crop.shape[0],img_crop.shape[1])
    image.img_data = img_crop
    return image.img_data

def rotate_image(image, degrees):
    pass

def flip_image(image, axis):
    image.img_data[0][0] = np.flip(image.img_data[0][0], axis=axis)
    return image.img_data
