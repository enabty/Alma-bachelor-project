import keras

def get_loss_function(loss_function):
    if loss_function == 'categorical_crossentropy':
        return keras.losses.categorical_crossentropy
    elif loss_function == 'binary_crossentropy':
        return keras.losses.binary_crossentropy
    elif loss_function == 'mean_squared_error':
        return keras.losses.mean_squared_error
    elif loss_function == 'mean_absolute_error':
        return keras.losses.mean_absolute_error
    elif loss_function == 'mean_absolute_percentage_error':
        return keras.losses.mean_absolute_percentage_error
    elif loss_function == 'mean_squared_logarithmic_error':
        return keras.losses.mean_squared_logarithmic_error
    elif loss_function == 'squared_hinge':
        return keras.losses.squared_hinge
    elif loss_function == 'hinge':
        return keras.losses.hinge
    elif loss_function == 'categorical_hinge':
        return keras.losses.categorical_hinge
    elif loss_function == 'logcosh':
        return keras.losses.logcosh
    elif loss_function == 'sparse_categorical_crossentropy':
        return keras.losses.sparse_categorical_crossentropy
    elif loss_function == 'binary_crossentropy':
        return keras.losses.binary_crossentropy
    elif loss_function == 'kullback_leibler_divergence':
        return keras.losses.kullback_leibler_divergence
    elif loss_function == 'poisson':
        return keras.losses.poisson
    elif loss_function == 'cosine_proximity':
        return keras.losses.cosine_proximity
    elif loss_function == 'is_categorical_crossentropy':
        return keras.losses.is_categorical_crossentropy
    else:
        print('Loss function not recognized. Using categorical_crossentropy')
        return keras.losses.categorical_crossentropy
    

def get_optimizer(optimizer):
    if optimizer == 'SGD':
        return keras.optimizers.SGD()
    elif optimizer == 'RMSprop':
        return keras.optimizers.RMSprop()
    elif optimizer == 'Adagrad':
        return keras.optimizers.Adagrad()
    elif optimizer == 'Adadelta':
        return keras.optimizers.Adadelta()
    elif optimizer == 'Adam':
        return keras.optimizers.Adam()
    elif optimizer == 'Adamax':
        return keras.optimizers.Adamax()
    elif optimizer == 'Nadam':
        return keras.optimizers.Nadam()
    else:
        print('Optimizer not recognized. Using Adam')
        return keras.optimizers.Adam()
    
def get_metrics(metrics):
    if metrics == 'accuracy':
        return keras.metrics.accuracy()
    elif metrics == 'binary_accuracy':
        return keras.metrics.binary_accuracy()
    elif metrics == 'categorical_accuracy':
        return keras.metrics.categorical_accuracy()
    elif metrics == 'sparse_categorical_accuracy':
        return keras.metrics.sparse_categorical_accuracy()
    elif metrics == 'top_k_categorical_accuracy':
        return keras.metrics.top_k_categorical_accuracy()
    elif metrics == 'sparse_top_k_categorical_accuracy':
        return keras.metrics.sparse_top_k_categorical_accuracy()
    else:
        print('Metric not recognized. Using accuracy')
        return keras.metrics.accuracy()