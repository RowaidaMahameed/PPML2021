import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import numpy
import matplotlib as pyplot
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import KFold
from tensorflow.python.keras import Sequential
import tensorflow
import tenseal as ts

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    return model
    
def enc_x(x_unenc):
    # parameters
    poly_mod_degree = 4096
    coeff_mod_bit_sizes = [40, 20, 40]
    # create TenSEALContext
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    # scale of ciphertext to use
    ctx_eval.global_scale = 2 ** 20
    # this key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()
    enc_x_ = [ts.ckks_tensor(ctx_eval, x.tolist()) for x in x_unenc]
    return enc_x_
    
def evaluate_model(X_train, y_Train, n_folds=5):
    accuracy, data = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for x_train, x_test in kfold.split(X_train):
        # create model
        model = create_model()
        trainX, trainY, testX, testY = X_train[x_train], y_Train[x_train], X_train[x_test], y_Train[x_test]
        enc_x_train = enc_x(trainX)
        enc_x_test = enc_x(testX)
        enc_y_train = enc_x(trainY)
        enc_y_test = enc_x(testY)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit model
        data_fit = model.fit(enc_x_train , enc_y_train, validation_data=(enc_x_test, enc_y_test), epochs=10, batch_size=32)
        testX = decrypt(enc_x_test)
        testY = decrypt(enc_y_test)
        
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)

        # stores Accuracy
        accuracy.append(acc)
        data.append(data_fit)

    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (numpy.mean(accuracy) * 100, numpy.std(accuracy) * 100, len(accuracy)))

    return accuracy, data   

if __name__ == "__main__":
    accuracy, data = evaluate_model(X_train, y_train)
    print ('The accuracy is :%.3f', % (numpy.mean(accuracy) * 100)
    

