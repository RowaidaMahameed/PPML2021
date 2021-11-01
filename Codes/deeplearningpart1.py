
#model creating and training:
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

#          load and prepare the data..
#load & split data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# One hot Code
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# convert from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range [0,1]
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



def evaluate_model(X_train, y_Train, n_folds=5):
    accuracy, data = list(), list()
    # prepare 5-cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for x_train, x_test in kfold.split(X_train):
        # create model
        model = create_model()
        # select rows for train and test
        trainX, trainY, testX, testY = X_train[x_train], y_Train[x_train], X_train[x_test], y_Train[x_test]

        #compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit model
        data_fit = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)

        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)

        # stores Accuracy
        accuracy.append(acc)
        data.append(data_fit)

    # serialize model to JSON and save the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save("final_model.h5")

    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (numpy.mean(accuracy) * 100, numpy.std(accuracy) * 100, len(accuracy)))

    return accuracy, data



# summarize model performance

def summarize_performance(acc):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (numpy.mean(acc) * 100, numpy.std(acc) * 100, len(acc)))

    # box and whisker plots of results
    pyplot.boxplot(acc)
    pyplot.show()



if __name__ == "__main__":
    evaluate_model(X_train, y_train)




# model loading and prediction..:
'''
import cv2
from tensorflow import keras
img = cv2.imread('2.png')

image = img.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)[1]
image = cv2.resize(image, (28, 28))
# display_image(image)
image = image.astype('float32')
image = image.reshape(1, 28, 28, 1)
image /= 255

# plt.imshow(image.reshape(28, 28), cmap='Greys')
# plt.show()
model = keras.models.load_model('final_model.h5')

pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

print("Predicted Number: ", pred.argmax())
'''



















