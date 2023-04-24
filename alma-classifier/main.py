from alma_classifier import pipeline_tensorflow
from alma_classifier.image_processing.manual_sorting import predict_fits
from keras.models import load_model
from alma_classifier.image_processing.image_augmentation import generate_pos_dataset
from alma_classifier.image_processing.manual_sorting import sort_manually, save_data_to_npy
from alma_classifier.image_processing.pre_processing import init_training_data_from_folder
import numpy as np
import shutil


"""

Note that the output of generate_pos_training_data() is saved to a .npy file and is 
400x400 so you need to run either crop_middle_100x100 or linnear_transformation before
you can use the data in the CNN

Example:

pos_data = [crop_middle_100x100(file) for file in pos_data if file.shape == (400, 400)]
pos_data = np.array([linear_transformation(file) for file in pos_data for i in range(augment_factor) if file.shape == (400, 400)])

"""


def generate_pos_training_data(load_directory='alma-classifier/data/fits/pos', save_directory='C:/ChalmersWorkspaces/KandidatArbete/raw_data/npy_temp/pos_temp.npy'):
    pos_data = generate_pos_dataset(load_directory)
    pos_data = sort_manually(pos_data)
    save_data_to_npy(pos_data, save_directory)

"""

Note that the aoutput of generate_neg_training_data() is saved to a .npy file and is ready to be 
used in the CNN directly

"""

def generate_neg_training_data(directory='alma-classifier/data/fits/neg'):
    neg_data = init_training_data_from_folder(directory)
    neg_data = sort_manually(neg_data)
    save_data_to_npy(neg_data, 'C:/ChalmersWorkspaces/KandidatArbete/raw_data/npy_train/neg_temp.npy')



def classify_data(file_paths, model, save_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg_fits/'):
    pos_image = predict_fits(file_paths, model)
    [shutil.copy2(name, 'C:/ChalmersWorkspaces/KandidatArbete/raw_data/pos') for (data, name) in pos_image]
    return pos_image


def create_and_save_neural_network(pos_npy_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/npy_train/pos_dataset.npy',
                                   neg_npy_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/npy_train/neg_dataset.npy', 
                                   save_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/CNN_model',
                                   lin_aug=False, 
                                   aug_factor=1):
    model = pipeline_tensorflow.pippeline_tensorflow(pos_npy_path, neg_npy_path, lin_aug, aug_factor)
    model.save(save_path)



def main(): 
    print('main')

    # create_and_save_neural_network(lin_aug=False, aug_factor=10)

    model = load_model('C:/ChalmersWorkspaces/KandidatArbete/raw_data/CNN_model')

    predict_fits('C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg_fits', model)
    

    # generate_pos_training_data()

    # model = pipeline_tensorflow.pippeline_tensorflow(linnear_aug=True, augmentation_factor=10)

    # model.save('test_model')

    # model = load_model('test_model')

    # images_of_interest = predict_fits('C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg', model)

    # images_of_interest = predict_fits('alma-classifier/data/fits/pos', model)





if __name__ == '__main__': main()
