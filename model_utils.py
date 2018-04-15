from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.layers import MaxPool2D
import os.path 

# define model
def define_model():
    model = Sequential()

    # 1st stage
    model.add(Conv2D(32, 3, input_shape=(48, 48, 1), padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # 2nd stage
    model.add(Conv2D(64, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # 3rd stage
    model.add(Conv2D(128, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding='same',
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.25))

    # FC layers
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation('softmax'))


    # compile the model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

    return model 

# load model weights
def model_weights(model):
    # load already saved model if needed
    if os.path.exists('models/weights.h5'):
        model.load_weights('models/weights.h5')
    else:
        print('No model to load !')
    return model

