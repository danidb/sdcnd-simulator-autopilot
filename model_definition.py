# The model applied in the exercise is defined herein.
#
# It is based on InceptionV3, a sufficiently efficient network for
# the hardware availalbe to me at the time of writing. The InceptionV3
# network, with pre-trained weights, is available from Keras.
#

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten
from keras.layers.pooling import AveragePooling2D

from keras.regularizers import l2

def steering_model(input_shape, optimizer='adam', loss='mean_squared_error'):
    """ Define InceptionV3-based transfer learning model for steering angle prediction.

    Args:
    input_shape: Shape of the image being input to the model. A tuple of length 2.

    Returns:
    Compiled keras model for image regression to determine steering angle.
    """

    mod = inception_model(input_shape = input_shape)

    mod.compile(optimizer=optimizer,
                loss=loss)

    return mod



def inception_model(input_shape=(48, 48, 3)):
    """ Define an inceptionV3-based architecture for image regression.

    The purpose of this function is solely to encapsulate the model architecture.

    Args:
    input_shape: Tuple of length 3, the shape of the model input. Must have width, height
    greater than 48, and 3 color channels.

    Returns:
    Keras architecture for regression, based on InceptionV3.
    """

    mod = Sequential()
    mod.add(VGG16(weights='imagenet', include_top=False, input_shape=input_shape))

    # For include_top = False, no classification block is included, it is replced here
    # to give us back some nice regression.
    mod.add(Flatten())
    mod.add(Dense(128, W_regularizer=l2(0.01), activation='relu'))
    mod.add(Dense(16, W_regularizer=l2(0.01), activation='tanh'))
    mod.add(Dense(1))

    return mod
