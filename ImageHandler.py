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
    def __init__(self, file_folder, file_name, file_format, img_data):
        self._file_folder = file_folder
        self._file_name = file_name
        self._file_format = file_format
        self._full_path = self.file_folder + self.file_name + self.file_format
        self._img_data = img_data
               
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
    
    # Augmentations #
    # Only meant for .fits files #

    def change_contrast(self, multiplicand = 1, addend = 0):
        if (self.file_format != ".fits"):
            print("file must be .fits format")
            return
        self.img_data = (self.img_data * multiplicand) + addend

    def noise_jitter(self, percetage_changed=1.0):
        if (self.file_format != ".fits"):
            print("file must be .fits format")
            return
        
        data = self.img_data[0][0]
        indices = np.asarray([(i,j) for i in range(len(data)) for j in range(len(data[i]))])
        np.random.shuffle(indices)
        n = int(np.floor(data.shape[0]*data.shape[1]*percetage_changed))
        for i in range(n):
            x = indices[i][0]
            y = indices[i][1]
            data[x][y] *= random.uniform(0.97,1.03)
    
    def neighbour_jitter(self, percetage_changed=1.0):
        data = self.img_data[0][0]
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
    
        
    
    # Cut out a square of a radius at a center = (x,y) and scale up the size with interpolation
    def crop_resize(self, radius, center):
        if (self.file_format != ".fits"):
            print("file must be .fits format")
            return

        matrix = self.img_data[0][0]
        mini = np.nanmin(matrix)
        maxi = np.nanmax(matrix)
        # Radius cannot be bigger than the closest edge
        radius = min(radius, center[0], self.img_data.shape[2] - center[0], center[1], self.img_data.shape[3] - center[1])
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
        arr2.shape = (1,1,arr2.shape[0],arr2.shape[1])
        self.img_data = arr2

    def standard_crop(self, center):
        crop_size = units.Quantity((100, 100), units.pixel)
        img_crop = Cutout2D(self.img_data[0][0], center, crop_size).data
        img_crop.shape = (1,1,img_crop.shape[0],img_crop.shape[1])
        self.img_data = img_crop
    
    def rotate_image(self, degrees):
        pass

    def flip_image(self, axis):
        self.img_data[0][0] = np.flip(self.img_data[0][0], axis=axis)
    
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
        n2 = int(0.1*len(values))
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
                    val = j[0]
                    x, y = j[1]
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
        




def tester():    
    hdul = fits.open("I:/Github/ALMA/Code/data/train/" + "b335_2017_band6_0" + ".fits")
    data = hdul[0].data
    hdul.close()
    test = Observation("I:/Github/ALMA/Code/data/train/", "b335_2017_band6_0" , ".fits", data)
    test.file_name = 'test'
    print(test.full_path)
    test.save_image(save_format='.fits')
    test.display_image()
    test.full_path = ("I:/Github/ALMA/Code/data/train/", "b335_2017_band6_0" , ".fits")
    print(test.full_path)

#tester()
