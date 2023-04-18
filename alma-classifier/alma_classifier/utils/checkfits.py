'''A function to manually look through the fits files downloaded from ALMA Archive.'''

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
import glob
import os
import shutil


def checkfits(datadir):
    # Creating directories to put files in after controlling them
    pos_dir = 'pos'
    neg_dir = 'neg'
    trash_dir = 'trash'

    parent_dir = datadir

    path_pos = os.path.join(parent_dir, pos_dir)
    path_neg = os.path.join(parent_dir, neg_dir)
    path_trash = os.path.join(parent_dir, trash_dir)


    # Check if directory exists and create it otherwise
    if not os.path.isdir(path_pos):
        os.mkdir(path_pos)
        print("Directory '% s' created" % path_pos)

    if not os.path.isdir(path_neg):
        os.mkdir(path_neg)
        print("Directory '% s' created" % path_neg)

    if not os.path.isdir(path_trash):
        os.mkdir(path_trash)
        print("Directory '% s' created" % path_trash)


    # Loop through the fits files and determine yourself
    # Store your fits files to be checked in a subfolder of datadir, similar to the path below
    files = glob.glob('/Users/erikredmo/School/Bachelors/ALMA/Code/Erik/Alminer/data/org/*.fits')

    # Keep track of how many files you've looked through
    count = 0

    for file in files:
        file_name = os.path.basename(file)
        file_data = fits.getdata(file).squeeze()

        plt.figure()
        plt.imshow(file_data, cmap="CMRmap_r")
        plt.colorbar()
        plt.show()
        
        txt = input('Enter which class this image belongs to. p=positive, n=negative, t=trash. Answer: ')
        if txt == 'p':
            shutil.move(os.path.dirname(file + '/' + file_name), path_pos + '/' + file_name)
        elif txt == 'n':
            shutil.move(os.path.dirname(file + '/' + file_name), path_neg + '/' + file_name)
        elif txt == 't':
            shutil.move(os.path.dirname(file + '/' + file_name), path_trash + '/' + file_name)
        else:
            print('Incorrect input, not moving file from original directory.')

        count += 1 

        print("You've looked through ", count, " images.")


DATADIR = '/Users/erikredmo/School/Bachelors/ALMA/Code/Erik/Alminer/data'

checkfits(DATADIR)