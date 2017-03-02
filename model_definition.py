# This module encapsulates the model architecture, and
# the simple operations required to compile the model.
# Keras is a magnificently simple library for rapid prototyping
# of neural networks.

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Convolution2D, Dropout
from keras.layers.pooling import AveragePooling2D

from keras.regularizers import l2

def steering_model(input_shape, optimizer='adam', loss='mean_squared_error'):
    """ Define InceptionV3-based transfer learning model for steering angle prediction.

    Args:
    input_shape: Shape of the image being input to the model. A tuple of length 3.
    optimizer: The optimizer used to compile the Keras model. The default is an Adam optimizer.
    loss: Loss function. For this regression task, the default of 'mean_squared_error' is quite
    suitable.

    Returns:
    Compiled keras model for image regression to determine steering angle.
    """

    mod = nvidia_model(input_shape = input_shape)

    mod.compile(optimizer=optimizer, loss=loss)

    return mod


def nvidia_model(input_shape=(48, 48, 3)):
    """ A convolutional neural network basd on Bojarski et. al. 2016

    This simple efficient network can be rapidly trained for steering angle prediction.
    The architecture below is heavily based on the neural network in 'End to End Learning for Self Driving Cars'
    (Bojarski et. al. 2016). The model applied here includes additional subsampling, and further regularization/
    the inclusion of extra dropout layers to prevent overfitting, and 'memorization' of the training
    track.

    Args:
    input_shape: The shape of the input tensor, a tuple of length 3. (width, height, n_channels)

    Returns:
    Keras model for image regression.
    """

    mod = Sequential()
    mod.add(Convolution2D(4, 5, 5, input_shape = input_shape, activation='relu', border_mode='valid'))
    mod.add(Dropout(0.1))
    mod.add(Convolution2D(8, 5, 5, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Dropout(0.1))
    mod.add(Convolution2D(16, 5, 5, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Dropout(0.1))
    mod.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Dropout(0.1))
    mod.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Flatten())
    mod.add(Dense(1164, W_regularizer=l2(0.01)))
    mod.add(Dense(100, W_regularizer=l2(0.01)))
    mod.add(Dense(50, W_regularizer=l2(0.01)))
    mod.add(Dense(10, W_regularizer=l2(0.01)))
    mod.add(Dense(1))

    return mod
