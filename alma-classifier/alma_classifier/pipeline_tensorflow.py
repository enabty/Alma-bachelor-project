from .models.tensorflow.model_01.data_handler import tensorflow_data_handler
from .models.tensorflow.model_01.evaluation import evaluate_model
from .image_processing.pre_processing import init_training_data_from_npy
from .models.tensorflow.model_01.model import model 
from .image_processing.manual_sorting import predict_fits


def pippeline_tensorflow(pos_npy_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/pos_dataset.npy', 
                         neg_npy_path='C:/ChalmersWorkspaces/KandidatArbete/raw_data/neg_dataset.npy',
                         linnear_aug=False, augmentation_factor=5):

    #-------------------Importing data-------------------#

    X, y = init_training_data_from_npy(pos_npy_path, neg_npy_path, linnear_aug, augmentation_factor)

    #-------------------Reshape to tensor-------------------#

    X_train, X_test, y_train, y_test = tensorflow_data_handler(X, y)

    #-------------------Creat model-------------------#

    nn_model = model()

    #-------------------Compile model-------------------#
    
    evaluate_model(X_train, X_test, y_train, y_test, nn_model)

    #-------------------Save model-------------------#
    # nn_model.save('C:/ChalmersWorkspaces/KandidatArbete/models/tensorflow/model_01/model.h5')
     

    return nn_model


__name__ == '__main__' and pippeline_tensorflow()
