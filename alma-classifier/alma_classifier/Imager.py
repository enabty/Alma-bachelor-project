import augmentation_methods as augm
import pandas as pd
import csv
import os
import numpy as np
import random
import glob
import sys
import queue
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


class Observation:
    def __init__(self, file_folder, file_name, file_format, img_data, aug_history):
        self._file_folder = file_folder
        self._file_name = file_name
        self._file_format = file_format
        self._full_path = self.file_folder + self.file_name + self.file_format
        self._img_data = img_data
        self._aug_history = aug_history
               
    # Properties #

    @property 
    def file_folder(self):
        return self._file_folder
    
    @file_folder.setter
    def file_folder(self, new_folder):
        self._file_folder = new_folder
        self.full_path = (new_folder,self.file_name,self.file_format)
    
    @property 
    def file_name(self):
        return self._file_name
    
    @file_name.setter
    def file_name(self, new_name):
        self._file_name = new_name
        self.full_path = (self.file_folder,new_name,self.file_format)

    @property 
    def file_format(self):
        return self._file_format
    
    @file_format.setter
    def file_format(self, new_format):
        self._file_format = new_format
        self.full_path = (self.file_folder,self.file_name,new_format)

    @property 
    def full_path(self):
        return self._full_path
    
    @full_path.setter
    def full_path(self, new_full_path):        
        if (self.file_folder != new_full_path[0]):
            self.file_folder = new_full_path[0]
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

    @property
    def aug_history(self):
        return self._aug_history
    
    @aug_history.setter
    def aug_history(self, new_history):
        self._aug_history = new_history
    
    # Methods #

    def display_image(self):
        if (self.file_format == '.png'):
            plt.imshow(self.img_data)
            plt.show()
        elif (self.file_format == '.fits'):          
            zscale = ZScaleInterval(contrast=0.25, nsamples=1)
            #values = values[np.isfinite(values)]


            #plt.imshow((self.img_data).squeeze(), origin='lower', cmap='rainbow', vmin=vmin, vmax=vmax)
            plt.imshow(zscale(self.img_data).squeeze(), origin='lower', cmap='rainbow')
            plt.show()
        else: 
            raise Exception('File format not supported')
    
    def save_image(self,save_folder = None, save_name = None, save_format = None):
        if (save_folder is None):
            save_folder = self.file_folder
        if (save_name is None):
            save_name = self.file_name
        if (save_format is None):
            save_format = self.file_format
        
        save_path = save_folder + save_name + save_format

        if (save_format == '.png'): 
            if (self.file_format == '.png'):
                plt.imsave(save_path, self.img_data)
            elif (self.file_format == '.fits'): 
                zscale = ZScaleInterval(contrast=0.25, nsamples=1)
                focus = np.nan_to_num(self.img_data)
                plt.imsave(save_path, zscale(focus).squeeze(), origin='lower', cmap='rainbow')
            else:
                raise Exception('File format not supported')       
            

        elif (save_format == '.fits'):
            if (self.file_format == '.fits'):
                hdu = fits.PrimaryHDU(self.img_data)
                hdul = fits.HDUList([hdu])
                hdul.writeto(save_path, overwrite=True)            
            else:
                raise Exception('File format not supported')
                          
        else: 
            raise Exception('File format not supported')
        
    def display_stats(self, display_histogram = False):
        if (self.file_format != ".fits"):
            print("file must be .fits format")
            return
        print("Stats for: " + self.file_name)
        print('Min:', np.nanmin(self.img_data))
        print('Max:', np.nanmax(self.img_data))
        print('Mean:', np.nanmean(self.img_data))
        print('Stdev:', np.nanstd(self.img_data) , '\n')
        if (display_histogram):
            histogram = plt.hist(self.img_data.flatten(), bins='auto')
            plt.show()

    def find_object_pos(self):
        cd = ClustarData(path=self.full_path, group_factor=0)        
        if len(cd.groups) > 0:
            disk = cd.groups[0]
            bounds = disk.image.bounds
            x = (bounds[2] + bounds[3])/2
            y = (bounds[0] + bounds[1])/2
            return (x, y)
        else:
            print("No object found in {}".format(self.full_path))
            return None
    
    def find_peaks(self):
        data = self.img_data[0][0]
    
        data = np.nan_to_num(data)
        # Descending order by intensity value
        sorterad = sorted([(data[i][j],(j,i)) for i in range(len(data)) for j in range(len(data[i]))], key=lambda x: x[0], reverse=True)
        values = np.asarray(sorterad,dtype=object)
        # (visited bool, index in sorterad)
        visited = [[[False,len(sorterad)] for j in range(len(data[i]))] for i in range(len(data))]

        n1 = int(0.001*len(values))
        n2 = int(0.3*len(values))
        for i in range(len(sorterad)):
            x, y = sorterad[i][1]
            visited[y][x][1] = i
        groups = []
        for i in range(n1):
            val = sorterad[i][0]
            x, y = sorterad[i][1]
            if (not visited[y][x][0]):
                q = queue.Queue()
                q.put(sorterad[i])
                subgroup = []
                while(not q.empty()):
                    j = q.get()
                    #val, [x, y] = q.get()
                    val = j[0]
                    x,y = j[1]
                    if (visited[y][x][0]): continue
                    visited[y][x][0] = True
                    subgroup.append(j)                   
                    neighbours = [(max(0, y-1),x), (min(data.shape[0]-1, y+1),x), (y,max(0, x-1)), (y,min(data.shape[0]-1, x+1))]
                    for candidate in neighbours:
                        y, x = candidate
                        if (visited[y][x][1] <= n2): q.put((data[y][x],(x,y)))
                groups.append(subgroup)

        filtererad = np.zeros((data.shape[0],data.shape[1]))
        for group in groups:
            for point in group:
                val = point[0]
                x, y = point[1]
                filtererad[y][x] = val
        print("test")
        self.img_data[0][0] = filtererad
        print(len(groups))
        for group in groups:
            print(max(group, key=itemgetter(0)))
        return groups

    
    # Augmentations #
    # Only meant for .fits files #

    def change_contrast(self, multiplicand = 1, addend = 0):
        res = augm.change_contrast(self, multiplicand, addend)

        self.aug_history = self.aug_history + ["change_contrast"]
        return res

    def noise_jitter(self, percetage_changed=1.0):       
        res = augm.noise_jitter(self, percetage_changed)

        self.aug_history = self.aug_history + ["noise_jitter"]
        return res
    
    def neighbour_jitter(self, percetage_changed=1.0):
        res = augm.neighbour_jitter(self, percetage_changed)

        self.aug_history = self.aug_history + ["neighbour_jitter"]
        return res
    
        
    
    # Cut out a square of a radius at a center = (x,y) and scale up the size with interpolation
    def crop_resize(self, radius, center):
        res = augm.crop_resize(self, radius, center)

        self.aug_history = self.aug_history + ["crop_resize"]
        return res

    def standard_crop(self, center):
        res = augm.standard_crop(self, center)

        self.aug_history = self.aug_history + ["standard_crop"]
        return res
    
    def rotate_image(self, degrees):
        res = augm.rotate_image(self, degrees)

        self.aug_history = self.aug_history + ["rotate_image"]
        return res

    def flip_image(self, axis):
        res = augm.flip_image(self, axis)

        self.aug_history = self.aug_history + ["flip_image"]
        return res
    

def tester():    
    hdul = fits.open("C:/Users/jensc/Documents/GitHub/ALMA/Code/data/org/fits/pos/" + "b335_2017_band6_0" + ".fits", memmap=False)
    data = hdul[0].data
    hdul.close()
    test = Observation("C:/Users/jensc/Documents/GitHub/ALMA/Code/data/org/fits/pos/", "b335_2017_band6_0" , ".fits", data, [])
    test.file_name = 'test'
    print(test.img_data)
    test.flip_image(1)
    test.neighbour_jitter(0.1)

    print(test.aug_history)
    test.display_image()

#tester()

    