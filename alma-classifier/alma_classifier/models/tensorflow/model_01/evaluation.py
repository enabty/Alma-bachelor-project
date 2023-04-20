import keras


"""

    This function is used to evaluate a tensorflow CNN.

"""

def evaluate_model(X_train, X_test, y_train, y_test, model):

    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    fit_info = model.fit(X_train, y_train,
                         batch_size=32,
                         epochs=5,
                         verbose=1,
                         validation_data=(X_test, y_test))

    print(model.evaluate(X_test, y_test, verbose=0))


__name__ == '__main__' and print('model.py works!')







