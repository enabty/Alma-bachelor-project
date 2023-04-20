from models.tensorflow.model_01.data_handler import tensorflow_data_handler
from models.tensorflow.model_01.evaluation import evaluate_model
from image_processing.pre_processing import init_from_npy
from models.tensorflow.model_01.model import model 

def pippeline_tensorflow():

    #-------------------Importing data-------------------#

    X, y = init_from_npy(
        'C:/ChalmersWorkspaces/KandidatArbete/raw_data/pos_dataset.npy',
        'C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg_dataset.npy')

    #-------------------Reshape to tensor-------------------#

    X_train, X_test, y_train, y_test = tensorflow_data_handler(X, y)

    #-------------------Creat model-------------------#

    cnn_model = model()

    #-------------------Compile model-------------------#
    
    evaluate_model(X_train, X_test, y_train, y_test, cnn_model)


__name__ == '__main__' and pippeline_tensorflow()
