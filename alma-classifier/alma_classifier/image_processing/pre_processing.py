from astropy.io import fits
import glob


def crop_around_middle_50x50_percent(fits): return fits[(int(fits.shape[0]*.25)):(int(fits.shape[0]*.75)),
                                                        (int(fits.shape[1]*.25)):(int(fits.shape[1]*.75))]


def pre_processing(file_path):
    print('pre_processing.py is working')
    # Load the data, pos_data is a list of matrices from a folder of fits files
    processed_img = [fits.getdata(file).squeeze() for file in glob.glob(file_path + '/*.fits')]

    # Reshape the data to be the middle 50% of the image,
    # This ensures that no extreme edge values are included where noise to intense
    return [crop_around_middle_50x50_percent(file) for file in processed_img if file.shape[0] > 500 and file.shape[1] > 500]


"""

Improvements:

We return a 500x500 2D-arr but since many FITS files are less than 500x500 we siply discard those in the pre-processing step.
To incluide more we should enlarge images smaller than that or return smaller images as standard.


"""

__name__ == '__main__' and print('pre_processing.py is working')
