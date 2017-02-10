from preprocess import prepare_training_files, generate_model_data, image_preprocess
import numpy as np
import os
import tensorflow
import keras
from scipy import misc

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, merge
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

#####
# Model Definition
#####
def simception_module(input_layer):
    """ An inception module for steering angle prediction

    An inception module implemented with the functional API provided by
    Keras, for the determination of steering angle from road images.

    See 'simception_model(...)' for a description of the architecture.

    Args:
    input_layer: Input to the module, Keras layer.

    Returns:
    Output layer of the inception module. Note that this is a concanetnation of multiple layers.
    """

    conv1x1 = Convolution2D(8, 1, 1, border_mode='same', activation='elu')(input_layer)
    conv5x5 = Convolution2D(8, 5, 5, border_mode='same', activation='elu')(conv1x1)
    conv3x3 = Convolution2D(8, 3, 3, border_mode='same', activation='elu')(conv1x1)

    output = merge([conv5x5, conv3x3], mode='concat')

    return output

def simception_model(model_input):
    """ Full simception model for determination of steering angle

    The architecture is as follows:

    Args:
    model_input: Keras model input layer.

    Returns:
    Model output - predicted steering angle (scaled)
    """
    conv_init = Convolution2D(4, 5, 5, activation='elu', border_mode='valid')(model_input)

    simception_one = simception_module(conv_init)
    dropout_one    = Dropout(0.5)(simception_one)

    simception_two = simception_module(dropout_one)
    dropout_two    = Dropout(0.5)(simception_two)

    simception_three = simception_module(dropout_two)
    dropout_three    = Dropout(0.5)(simception_three)


    flattened = Flatten()(dropout_three)

    fully_connected_one   = Dense(256, activation='elu', W_regularizer=l2(0.01))(flattened)
    fully_connected_two   = Dense(128, activation='elu', W_regularizer=l2(0.01))(fully_connected_one)
    fully_connected_three = Dense(16, activation='elu', W_regularizer=l2(0.01))(fully_connected_two)
    fully_connected_final = Dense(1, activation='linear')(fully_connected_three)

    return fully_connected_final


#####
# Data loading and preparation
#####

# We create a training set, and a validation set with a 70-30 split.
print("Preparing training data.")
training_nsamples, validation_nsamples = prepare_training_files("./data/driving_log.csv")

print()
print("_________________________________________")
print("Training, N: ", training_nsamples, " Validation, N: ", validation_nsamples)


print("Data prepared.")
# Model prep for trainingco
model_input = Input(shape=(40, 80, 1))
model_final = Model(input=model_input, output=simception_model(model_input))

model_final.compile(optimizer='adam', loss='mean_squared_error')


batch_size = 128
print("Batch size: ", batch_size)
print("_________________________________________")
model_final.fit_generator(generate_model_data('./data/training_log.csv',
                                              image_preprocess,
                                              batch_size=batch_size,
                                              nsamples=training_nsamples),
                          validation_data=generate_model_data('./data/validation_log.csv',
                                                              image_preprocess,
                                                              batch_size=batch_size,
                                                              nsamples=validation_nsamples),
                          samples_per_epoch=(training_nsamples // batch_size)*batch_size,
                          nb_val_samples=(validation_nsamples // batch_size)*batch_size,
                          nb_epoch=10,
                          verbose=1)
model_final.save('model.h5', overwrite=True)
print("_________________________________________")
print("Model saved.")
print("_________________________________________")
