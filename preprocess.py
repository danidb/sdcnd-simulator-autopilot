import cv2
import numpy as np
import os
import csv
import random
from scipy import misc
from itertools import islice
import sklearn

if int(sklearn.__version__.replace('.', '')) >= 180 :
    from sklearn.model_selection import train_test_split as tts
else:
    from sklearn.cross_validation import train_test_split as tts
GROW = None

def image_preprocess(img):
    """ Pre-process images from the udacity SDCND simulator

    Args:
    img: Image to process. Must be in color (RGB).

    Returns:
    The input img after application of (1) grayscale ...
    """

    image_working = img
    image_working = cv2.cvtColor(image_working, cv2.COLOR_RGB2GRAY)
    image_working = cv2.resize(image_working, (40, 80), interpolation=cv2.INTER_AREA)
    image_working = clahe_GRAY(image_working, clipLimit=1, tileGridSize=(4,4))

    return image_working.reshape(40, 80, 1)



def split_log_row(log_row, log_path):
    """ Split one row of the driving log into three rows, one per camera.

    Args:
    log_row: Row of the log file
    log_path: Path of the log file

    Returns:
    List of length three, each entry is a row of the expanded log file.
    """

    expand_image_path = lambda img_path: os.path.abspath(os.path.join(os.path.dirname(log_path), img_path))

    c_img_path, l_img_path, r_img_path, angle, _, _, _ = log_row

    return [[expand_image_path(c_img_path.strip()), float(angle.strip())]]
#            [expand_image_path(l_img_path.strip()), float(angle.strip())],
#            [expand_image_path(r_img_path.strip()), float(angle.strip())]]


def process_log_row(log_row, img_process_FUN):
    """ Process a row of the driving log.

    For the purposes of this experiment, we are only concerned with centre image, and
    the steering angle.

    Args:
    log_row: One row of the log driving log.
    img_process_FUN: Function to apply to each loaded image, preparing it for input to the pipeline.

    Returns:
    A tuple (A, B) where A is a numpy array with the three images, and B is a numpy array
    with the steering angle repeated once per image.
    """

    # We'll be applying this method over a list of image paths
    img_path, angle = log_row

    image = img_process_FUN(misc.imread(img_path))
    angle = float(angle)

    return (image, angle)


def clahe_GRAY(img, clipLimit, tileGridSize):
    """ Apply Contrast-Limited Adaptive Histogram Equalization with OpenCV

    Contrast-Limited Adaptive Histogram Equalization is applied to a grayscale image.

    Args:
        img: Input image, should be in RGB colorspace.
        clipLimit: Passed to cv2.createCLAHE
        tileGridSize: Passed to cv2.createCLAHE

    Returns:
        The input image, with CLAHE applied, still as grayscale.
    """
    img_clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_ret = img_clahe.apply(img)
    return(img_ret)

def prepare_training_files(log_path):
    """ Shuffle and split the driving logfile into training and validation data

    Args:
    log_path: Path to the original log file output by the simulator.

    Returns:
    An integer, the nubmer of validation samples.
    Also creates two CSV files - training.csv and validation.csv, in the same directory as log_path.
    The union of these files contains all the rows in log_path.
    """

    with open(log_path, 'rt') as orig_logfile:
        # Note that we skip the first line, as it only contains row headers.
        orig_log = list(csv.reader(orig_logfile))[1:]
        orig_log = [split_log_row(log_row, log_path) for log_row in orig_log]
        orig_log = [row for entry in orig_log for row in entry]

        random.shuffle(orig_log)

        training_log, validation_log = tts(orig_log, random_state=1729, test_size=0.3)

    with open(os.path.dirname(log_path) + '/training_log.csv', 'wt') as training_logfile:
        csv.writer(training_logfile, quoting=csv.QUOTE_NONE, delimiter=",").writerows(training_log)

    with open(os.path.dirname(log_path) + '/validation_log.csv', 'wt') as validation_logfile:
        csv.writer(validation_logfile, quoting=csv.QUOTE_NONE, delimiter=",").writerows(validation_log)

    return (len(training_log), len(validation_log))


def generate_model_data(expanded_log_path, img_process_FUN, batch_size, nsamples):
    """ Generator for training model training data

    Args:
    expanded_log_path: Relative path of expanded training/validation logs.
    image_process_FUN: Image pre-processing function to apply.
    batch_size: Number of elements to return with each iteration (model batch size).
    nsamples: Total number of samples (obtained when the original driving log is parsed).

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

                        image, angle = process_log_row(row, img_process_FUN)

                        images += [image]
                        angles += [angle]

                    batch = []

                    yield (np.array(images), np.array(angles))
