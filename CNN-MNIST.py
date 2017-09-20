from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12
image_channels = 1
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#There are two types of image channels: channels_first and channels_last which denotes the order of image change
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], image_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], image_channels, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, image_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, image_channels)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def cnn_model1():
    model = Sequential()
    '''
    Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, 
            dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
    When using this layer as the first layer in a model, 
    provide the keyword argument input_shape (tuple of integers, does not include the sample axis),
     e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in  data_format="channels_last"
    '''
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape= input_shape))
    model.add(Conv2D(64, kernel_size =  (3,3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

def cnn_model2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5),
                     activation='relu',
                     input_shape= input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(5,5),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

def cnn_model3():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5),activation='relu',
                     input_shape= input_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size= batch_size, epochs = epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

if __name__ == '__main__':
    cnn_model3()
