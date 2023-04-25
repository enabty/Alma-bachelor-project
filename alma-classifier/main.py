from alma_classifier import pipeline_tensorflow
from alma_classifier.image_processing.manual_sorting import predict_fits
from keras.models import load_model
from alma_classifier.image_processing.image_augmentation import generate_pos_dataset
from alma_classifier.image_processing.manual_sorting import sort_manually, save_data_to_npy
from alma_classifier.image_processing.pre_processing import init_training_data_from_folder
import numpy as np
import shutil

"""
*****//PATHS//******
"""


ORIGINAL_POS_FITS = 'alma-classifier/data/fits/pos'   
ORIGINAL_NEG_FITS = 'alma-classifier/data/fits/neg'

POS_TRAIN = 'C:/ChalmersWorkspaces/KandidatArbete/data/npy_train/pos_dataset.npy'
NEG_TRAIN = 'C:/ChalmersWorkspaces/KandidatArbete/data/npy_train/neg_dataset.npy'

CNN_MODEL = 'C:/ChalmersWorkspaces/KandidatArbete/data/CNN_model'

UNCLASSIFIED_FITS = 'C:/ChalmersWorkspaces/KandidatArbete/data/fits_to_classify'
CLASSIFIED_FITS = 'C:/ChalmersWorkspaces/KandidatArbete/data/classified_fits'


"""

Note that the output of generate_pos_training_data() is saved to a .npy file and is 
400x400 so you need to run either crop_middle_100x100 or linnear_transformation before
you can use the data in the CNN. When changing the input fits files you need to manually 
change the degreees in the rotate array so that asll tha gaussian discs line up and is 
compatible with being combined with eachother. 

Example:

pos_data = [crop_middle_100x100(file) for file in pos_data if file.shape == (400, 400)]
pos_data = np.array([linear_transformation(file) for file in pos_data for i in range(augment_factor) if file.shape == (400, 400)])

NOTE!: This is done in initinit_training_data_from_npy() in pre_processing.py which is called
in pippeline_tensorflow() in pipeline_tensorflow.py. Which is standard for the CNN.

"""


def generate_pos_training_data(load_directory=ORIGINAL_POS_FITS, save_directory=POS_TRAIN):
    pos_data = generate_pos_dataset(load_directory)
    pos_data = sort_manually(pos_data)
    save_data_to_npy(pos_data, save_directory)


"""

Note that the aoutput of generate_neg_training_data() is saved to a .npy file and is ready to be 
used in the CNN directly

"""

def generate_neg_training_data(load_directory=ORIGINAL_NEG_FITS):
    neg_data = init_training_data_from_folder(load_directory)
    neg_data = sort_manually(neg_data)
    save_data_to_npy(neg_data, NEG_TRAIN)


"""

Creates and traines a CNN and saves the trained CNN to specified folder.

"""

def train_CNN(pos_npy_path=POS_TRAIN,
                neg_npy_path=NEG_TRAIN,
                save_path=CNN_MODEL,
                lin_aug=False,
                aug_factor=1):
    model = pipeline_tensorflow.pippeline_tensorflow(
        pos_npy_path, neg_npy_path, lin_aug, aug_factor)
    model.save(save_path)

"""

Loads a trained CNN and uses it to classify the fits files in the specified folder.
The classified fits files are then saved to the specified folder.

"""

def classify_data(read_path = UNCLASSIFIED_FITS, model = CNN_MODEL, save_path=CLASSIFIED_FITS):
    pos_image = predict_fits(read_path, load_model(model))
    [shutil.copy2(name, save_path) for (data, name) in pos_image]


def main():
    print('main')

    classify_data()


    
if __name__ == '__main__': main()
