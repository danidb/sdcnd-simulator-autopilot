import cv2
import numpy as np
import os
import csv
import random
from scipy import misc
from itertools import islice
import sklearn
import progressbar


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

    image_working = image
    image_working = image_working[60:130,0:300,:]
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

n
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

                        image = misc.imread(row[0])
                        images += [image.reshape(input_shape)]
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
