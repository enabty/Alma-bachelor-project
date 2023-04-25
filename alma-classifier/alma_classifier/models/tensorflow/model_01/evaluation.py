import keras


"""

    This function is used to evaluate a tensorflow CNN.

"""

def evaluate_model(X_train, X_test, y_train, y_test, model, 
                   loss_function='binary_crossentropy' , learning_rate=0.0001, optimizer='Adam', metrics='accuracy'):

    model.summary()

    model.compile(loss = keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    fit_info = model.fit(X_train, y_train,
                         batch_size=2,
                         epochs=30,
                         verbose=1,
                         validation_data=(X_test, y_test))

    print(model.evaluate(X_test, y_test, verbose=0))


__name__ == '__main__' and print('model.py works!')
