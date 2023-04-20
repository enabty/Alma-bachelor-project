import numpy as np
import tensorflow as tf

from keras import utils as np_utils
from sklearn.model_selection import train_test_split


"""

    This function is used to reshape the data to a tensorflow format.

"""


def tensorflow_data_handler(X, y):

    X = np.array([tf.convert_to_tensor(fits) for fits in X])
    y = np.array([tf.convert_to_tensor(fits) for fits in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    return X_train, X_test, y_train, y_test


__name__ == '__main__' and print('data_handler.py works!')


