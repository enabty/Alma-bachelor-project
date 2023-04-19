from astropy.io import fits
import glob
import os
from Imager import Observation


def crop_around_middle_50x50_percent(fits): return fits[(int(fits.shape[0]*.25)):(int(fits.shape[0]*.75)),
                                                        (int(fits.shape[1]*.25)):(int(fits.shape[1]*.75))]


def pre_processing(folder_path):
    print('pre_processing.py is working')
    # Load the data, pos_data is a list of matrices from a folder of fits files

    processed_img = []
    for f in os.listdir(folder_path):
        file_name, file_format = os.path.splitext(os.path.basename(f))
        if (file_format != '.fits'): continue
        hdul = fits.open(folder_path + f, memmap=False)
        data = hdul[0].data
        hdul.close()
        
        image = Observation(folder_path, file_name, file_format, data, [])

    # Reshape the data to be the middle 50% of the image,
    # This ensures that no extreme edge values are included where noise to intense
    for image in processed_img:
        if (image.shape[0] > 500 and image.shape[1] > 500):
            image.crop_middle(0.5)

    return processed_img


"""

Improvements:

We return a 500x500 2D-arr but since many FITS files are less than 500x500 we siply discard those in the pre-processing step.
To incluide more we should enlarge images smaller than that or return smaller images as standard.


"""

__name__ == '__main__' and print('pre_processing.py is working')
