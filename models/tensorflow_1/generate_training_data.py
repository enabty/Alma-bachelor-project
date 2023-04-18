xfrom src.image_procesing.pre_processing import pre_processing
from src.image_procesing.image_augmentation import image_augmentation
# save numpy array as csv file
import numpy as np
from numpy import asarray
from numpy import savetxt

import pandas as pd


def generate_training_data(gen_data=bool):
    if gen_data == True:
        print('Generating training data')

        pos_data = pre_processing(
            'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\pos_fits')
        pos_data = image_augmentation(pos_data)

        # pos_data = np.reshape(pos_data[0], -1)
        pos_data = np.asarray(pos_data, dtype=np.ndarray)

        # np.savetxt('training_data.csv', pos_data)

        pd.DataFrame(pos_data).to_csv('training_data.csv', index=False)

        neg_data = pre_processing(
            'C:\ChalmersWorkspaces\KandidatArbete\Alma-bachelor-project\data\neg_fits')
        neg_data = image_augmentation(neg_data)
        savetxt('training_data.csv', np.asarray(neg_data), delimiter=',')

        # neg_data = np.reshape(neg_data[0], -1)
        # neg_data = np.asarray(neg_data)



__name__ == '__main__' and generate_training_data(True)
