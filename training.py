from preprocess import prepare_training_files, generate_model_data, image_preprocess
import numpy as np
import os
import tensorflow as tf
import keras
import csv
import argparse

from scipy import misc

import model_definition as md

#####
# Model Definition
#####
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
    input_shape = (48, 48, 3)

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


        batch_size = 256

        print("Batch size: ", batch_size)
        print("_________________________________________")

        train_steering('./data/training_log.csv', './data/validation_log.csv',
                       training_nsamples, validation_nsamples,
                       input_shape=input_shape, epochs=2)

        print("_________________________________________")
        print("Model saved.")
        print("_________________________________________")
