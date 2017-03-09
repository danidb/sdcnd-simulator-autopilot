import numpy as np
import os
import tensorflow as tf
import keras
import csv
import argparse
import cv2
import csv
import random
from scipy import misc
from itertools import islice
import sklearn
import progressbar

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Convolution2D, Dropout, Cropping2D
from keras.layers.core import Lambda
from keras.layers.pooling import AveragePooling2D
from keras.regularizers import l2


if int(sklearn.__version__.replace('.', '')) >= 180 :
    from sklearn.model_selection import train_test_split as tts
else:
    from sklearn.cross_validation import train_test_split as tts

def image_preprocess(image, input_shape):
    """ Pre-process images from the udacity SDCND simulator

    Args:
    image: Image to process. Must be in color (RGB).
    input_shape: The final shape to which the input image is resized.

    Returns:
    The input image after cropping, the aplication of CLAHE, and resizing.
    """
    image_working = image[40:150,:,:]
    image_working = cv2.resize(image_working, (input_shape[0], input_shape[1]))
    image_working = clahe_RGB(image_working, clipLimit=2, tileGridSize=(2,2))
    image_working = cv2.cvtColor(image_working, cv2.COLOR_RGB2YUV)

    return image_working

def split_log_row(log_row, log_path, steering_correction):
    """ Split one row of the driving log into three rows, one per camera.

    Args:
    log_row: Row of the log file
    log_path: Path of the log file
    steering_correction: Correction applied to angle coming from left/right cameras.

    Returns:
    List of length three, each entry is a row of the expanded log file.
    """

    expand_image_path = lambda image_path: os.path.abspath(os.path.join(os.path.dirname(log_path), image_path))

    c_image_path, l_image_path, r_image_path, angle, _, _, _ = log_row

    return [[expand_image_path(c_image_path.strip()), float(angle.strip())],
            [expand_image_path(l_image_path.strip()), float(angle.strip()) + steering_correction],
            [expand_image_path(r_image_path.strip()), float(angle.strip()) - steering_correction]]

def preprocess_log_row(log_row, image_process_FUN, input_shape):
    """ Process a row of the driving log.

    For the purposes of this experiment, we are only concerned with the steering angle.
    Images are pre-processed. They are saved in the same directory. Pre-processed
    images are given a leading 'p_'.

    Args:

    log_row: One row of the log driving log.

    image_process_FUN: Function to apply to each loaded image, preparing it for input to the pipeline.

    input_shape: Size to which the cropped image is to be resized.

    Returns:
    A tuple (A, B). A is the path of the pre-processed non-mirror image.
    B is the original steering angle.
    """

    image_path, angle = log_row
    image_dir, image_name = os.path.split(image_path)

    # The original image is preprocessed and saved, this pre-processed images is
    # then saved.
    image = image_process_FUN(misc.imread(image_path), input_shape=input_shape)

    angle = float(angle)

    p_image_path = os.path.join(image_dir, 'p_' + image_name)

    misc.imsave(p_image_path, image)

    return (p_image_path, angle)



def prepare_training_files(log_path, input_shape, steering_correction):
    """ Preprocess, shuffle, and split simulator data into training/validation sets.

    Args:

    log_path: Path to the original log file output by the simulator.

    input_shape: Final size to which the input image is resinzed, after cropping.

    steering_correction: Correction applied to images coming from the left/right cameras.

    Returns:
    An integer, the nubmer of validation samples.
    Also creates two CSV files - training.csv and validation.csv, in the same directory as log_path.
    The union of these files contains all the rows in log_path.
    """

    with open(log_path, 'rt') as orig_logfile:
        # Note that we skip the first line, as it only contains row headers.
        orig_log = list(csv.reader(orig_logfile))[1:]
        orig_log = [split_log_row(log_row, log_path, steering_correction) for log_row in orig_log]
        orig_log = [row for entry in orig_log for row in entry]

        preproc_log = []
        pbar = progressbar.ProgressBar(max_value=len(orig_log))
        pbar.start()
        for i,row in enumerate(orig_log):

            p_image_path, angle  = preprocess_log_row(row,
                                                      image_process_FUN = image_preprocess,
                                                      input_shape = input_shape)
            preproc_log += [(p_image_path, angle)]

            pbar.update(i)

        pbar.finish()

        preproc_log = sklearn.utils.shuffle(preproc_log, random_state=1729)

        training_log, validation_log = tts(preproc_log, random_state=1729, test_size=0.2)

    with open(os.path.dirname(log_path) + '/training_log.csv', 'wt') as training_logfile:
        csv.writer(training_logfile, quoting=csv.QUOTE_NONE, delimiter=",").writerows(training_log)

    with open(os.path.dirname(log_path) + '/validation_log.csv', 'wt') as validation_logfile:
        csv.writer(validation_logfile, quoting=csv.QUOTE_NONE, delimiter=",").writerows(validation_log)

    return (len(training_log), len(validation_log))


def generate_model_data(expanded_log_path, input_shape, batch_size, nsamples):
    """ Generator for training model training data

    Args:
    expanded_log_path: Relative path of expanded training/validation logs.

    batch_size: Number of elements to return with each iteration (model batch size).

    input_shape: Shape if input, tuple of length 3.

    nsamples: Total number of samples (obtained when the input log is parsed).

    Returns:
    A tuple ([images,...], [angles,...]) where the length of both lists = batch_size.
    """
    while 1:
        with open(expanded_log_path, 'rt') as logfile:

            data_accumulator = ([], [])
            logfile_reader = csv.reader(logfile, delimiter=',')

            batch = []
            batch_count = 0
            samples_remaining = nsamples
            for batch_row in logfile_reader:

                if batch_count < batch_size and samples_remaining > 1:
                    batch = batch + [tuple(batch_row)]
                    batch_count += 1
                    samples_remaining -= 1

                else:
                    batch_count = 0

                    images = []
                    angles = []

                    for row in batch:

                        images += [misc.imread(row[0])]
                        angles += [float(row[1])]

                    batch = []

                    yield (np.array(images), np.array(angles))



def clahe_RGB(img, clipLimit, tileGridSize):
    """ Apply Contrast-Limited Adaptive Histogram Equalization with OpenCV

    Contrast-Limited Adaptive Histogram Equalization is applied to each
    of the three color channels of an RGB image. The result is returned
    as an RGB image.

    Args:
        img: Input image  should be in RGB colorspace.
        clipLimit: Passed to cv2.createCLAHE
        tileGridSize: Passed to cv2.createCLAHE

    Returns:
        The input image  with CLAHE applied  in RGB
    """

    r, g, b = cv2.split(img)

    img_clahe   = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe_r = img_clahe.apply(r)
    img_clahe_g = img_clahe.apply(g)
    img_clahe_b = img_clahe.apply(b)

    img_ret = cv2.merge((img_clahe_r,  img_clahe_g,  img_clahe_b))

    return(img_ret)

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


def nvidia_model(input_shape=(160, 320, 3)):
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

    normalizer = lambda img: img/255 - 0.5

    mod = Sequential()
    mod.add(Lambda(normalizer, input_shape=input_shape))
    mod.add(Convolution2D(4, 5, 5, input_shape=input_shape, activation='relu', border_mode='valid'))
    mod.add(Convolution2D(8, 5, 5, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Convolution2D(16, 3, 3, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Dropout(0.5))
    mod.add(Convolution2D(32, 3, 3, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Dropout(0.5))
    mod.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu', border_mode='valid'))
    mod.add(Flatten())
    mod.add(Dense(1024))
    mod.add(Dense(512))
    mod.add(Dense(256))
    mod.add(Dense(128))
    mod.add(Dense(32))
    mod.add(Dense(16))
    mod.add(Dense(1))

    return mod

def train_steering(training_logpath, validation_logpath, training_n, validation_n,
                   outpath='model.h5', input_shape=(48, 48, 3), batch_size=16, epochs=5):
    """ Train the steering angle model.

    Args:

    training_logpath: Path to training log produced during initial pre-processing.
    validation_logpath: Path to the validation log produced during initial pre-processing.

    training_n: Number of training samples.
    validation_n: Number of validation samples.

    outpath: Path to which the model weights etc. are saved.

    input_shape: Size of input provided to the model. Note that this must be
    a Tuple (width >= 48, height >= 48, 3).

    batch_size: Batch size for training. Something around 16 usually
    works without running out of memory.

    epochs: Number of training epochs. The default is 5.

    This method encapsulates the procedure for training the steering
    angle model.


    Return:
    Nothing. Saves the model to 'outpath'.
    """

    # Compute some required inputs to the fit_generator metod
    samples_per_epoch = (training_n // batch_size) * batch_size
    nb_val_samples = (validation_n // batch_size) * batch_size

    mod = md.steering_model(input_shape=input_shape,
                            optimizer='adam',
                            loss='mean_squared_error')


    training_generator = generate_model_data(training_logpath,
                                             input_shape = input_shape,
                                             batch_size = batch_size,
                                             nsamples = training_n)

    validation_generator = generate_model_data(validation_logpath,
                                               input_shape = input_shape,
                                               batch_size = batch_size,
                                               nsamples = validation_n)

    mod.fit_generator(training_generator,
                      validation_data = validation_generator,
                      samples_per_epoch = samples_per_epoch,
                      nb_val_samples = nb_val_samples,
                      nb_epoch = epochs)

    mod.save(outpath, overwrite=True)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model preparation")

    parser.add_argument(
        "--skip-preproc",
        nargs="?",
        dest="skip_preproc",
        const=True, default=False,
        help=("Flag. If present, we do not perform pre-processing, which may have already been"
              " performed perviously"))

    parser.add_argument(
        "--skip-training",
        nargs="?",
        dest="skip_training",
        const=True, default=False,
        help=("Flag. If present, skip model training. One may only want to preprocess the training"
              " data, without then training the model."))

    args = parser.parse_args()
    input_shape = (60, 60, 3)

    if not args.skip_preproc:
        print("Preparing training data.")
        training_nsamples, validation_nsamples = prepare_training_files(log_path="./data/driving_log.csv",
                                                                        input_shape=input_shape,
                                                                        steering_correction=0.1)
        print()
        print("_________________________________________")
        print("Training, N: ", training_nsamples, " Validation, N: ", validation_nsamples)


        print("Data preppared.")


    if not args.skip_training:
        with open('./data/training_log.csv', 'rt') as tl:
            training_reader = csv.reader(tl)
            training_nsamples = sum(1 for row in training_reader)

        with open('./data/validation_log.csv', 'rt') as vl:
            validation_reader = csv.reader(vl)
            validation_nsamples = sum(1 for row in validation_reader)


        batch_size = 32

        print("Batch size: ", batch_size)
        print("_________________________________________")

        train_steering('./data/training_log.csv', './data/validation_log.csv',
                       training_nsamples, validation_nsamples,
                       input_shape=input_shape, epochs=5)

        print("_________________________________________")
        print("Model saved.")
        print("_________________________________________")
