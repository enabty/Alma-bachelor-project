import pandas as pd
import csv
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy import units
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
import glob
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from clustar.core import ClustarData
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import random
from scipy.ndimage import rotate

class Observation:
    def __init__(self, file_path, file_name, file_format, img_data):
        self._file_path = file_path
        self._file_name = file_name
        self._file_format = file_format
        self._full_path = self.file_path + self.file_name + self.file_format
        self._img_data = img_data
               
    # Properties #

    @property 
    def file_path(self):
        return self._file_path
    
    @file_path.setter
    def file_path(self, new_path):
        self._file_path = new_path
        self.full_path = (new_path,self.file_name,self.file_format)
    
    @property 
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, new_name):
        self._file_name = new_name
        self.full_path = (self.file_path,new_name,self.file_format)

    @property 
    def file_format(self):
        return self._file_format
    
    @file_format.setter
    def file_format(self, new_format):
        self._file_format = new_format
        self.full_path = (self.file_path,self.file_name,new_format)

    @property 
    def full_path(self):
        return self._full_path
    
    @full_path.setter
    def full_path(self, new_full_path):        
        if (self.file_path != new_full_path[0]):
            self.file_path = new_full_path[0]
        if (self.file_name != new_full_path[1]):
            self.file_name = new_full_path[1]
        if (self.file_format != new_full_path[2]):
            self.file_format = new_full_path[2]
        self._full_path = new_full_path[0] + new_full_path[1] + new_full_path[2]
    
    @property
    def img_data(self):
        return self._img_data
    
    @img_data.setter
    def img_data(self, new_img_data):
        self._img_data = new_img_data
    
    # Methods #

    def display_image(self):
        if (self.file_format == '.png'):
            plt.imshow(self.img_data)
            plt.show()
        elif (self.file_format == '.fits'):
            zscale = ZScaleInterval(contrast=0.25, nsamples=1)
            plt.imshow(zscale(self.img_data).squeeze(), origin='lower', cmap='rainbow', aspect='auto')
            plt.show()
        else: 
            raise Exception('File format not supported')
    
    def save_image(self):
        if (self.file_format == '.png'):            
            plt.imshow(self.img_data)
            plt.axis('off')
            plt.ioff()
            plt.savefig(self.full_path, bbox_inches='tight', pad_inches=0)
        elif (self.file_format == '.fits'):
            n = np.ones((self.img_data.shape[2],self.img_data.shape[3]))
            n.shape = (1,1,n.shape[0],n.shape[1])
            hdu = fits.PrimaryHDU(n)
            hdul = fits.HDUList([hdu])
            hdul[0].data[0][0] = self.img_data
            hdul.writeto(self.full_path, overwrite=True)
        else: 
            raise Exception('File format not supported')


data = fits.getdata("I:/Github/ALMA/Code/data/train/" + "b335_2017_band6_0" + ".fits")
test = Observation("I:/Github/ALMA/Code/data/train/", "b335_2017_band6_0" , ".fits", data)
test.file_name = 'test'
print(test.full_path)
test.save_image()
test.display_image()
test.full_path = ("I:/Github/ALMA/Code/data/train/", "b335_2017_band6_0" , ".fits")
print(test.full_path)

