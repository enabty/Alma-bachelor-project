import pandas as pd
import csv
import os
import numpy as np
import random
import glob
import sys
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
from ImageHandler import Observation
import warnings



def tester():
    path = "C:/Users/jensc/Documents/Github/ALMA/Code/data/org/fits/neg/"
    for f in os.listdir(path):
        hdul = fits.open(path + f, memmap=False)
        data = hdul[0].data
        hdul.close()
        file_name, file_format = os.path.splitext(os.path.basename(f))
        test = Observation(path, file_name, file_format, data)
        if (test.file_name != "member.uid___A001_X2fe_X383.ari_l.MMS6_sci.spw0_1_2_3_224111MHz.12m.cont.I.pbcor_test"):
            continue
            pass
        #object_pos = test.find_object_pos()
        object_pos = None
        test.crop_resize(25,(50,45))
        test.display_image()    
        test.flip_image(1)
        
        test.display_image()
        test.file_name = test.file_name + '_test'
        if object_pos != None:
            values = np.asarray(test.img_data)
            np.set_printoptions(threshold=sys.maxsize)
            print(test.img_data.shape)
            #test.crop_resize(50,(int(object_pos[0]),int(object_pos[1])))
            test.standard_crop(object_pos)
        else:
            test.crop_resize(50,(test.img_data.shape[3]//2,test.img_data.shape[2]//2))
        
        #test.save_image(save_format='.png', save_folder='C:/Users/jensc/Documents/Github/ALMA/Code/data/org/png/neg/')
        #test.display_image()

  
tester()