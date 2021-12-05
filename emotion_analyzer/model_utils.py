# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Helper methods for Emotion model related work

Usage: python -m emotion_analyzer.model_utils

"""
# ===================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
import os.path

from emotion_analyzer.exceptions import ModelFileMissing 


def define_model():
    """Creates a model from a predefined architecture

    Returns:
        model: Sequential keras model
    """
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


def load_model_weights(model, model_path='./models/weights.h5'):
    """Loads trained model weights from model file.

    Args:
        model (keras model): [Untrained model with init weights]

    Returns:
        [keras model]: [Model loaded with trained weights]
    """
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        raise ModelFileMissing
    return model