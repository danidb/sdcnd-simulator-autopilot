# The source contained herein produces image output,
# for the evaluation of the pre-processing steps.
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import preprocess as pp
import os
from scipy import misc
from random import sample


def prepare_sample_images(training_log, image_size, out_dir='./sample_images', n=5):
    """ Plot n sample images, selected at random from the training data.

    Both 'before' and 'after' iamges are drawn to a given file, the naming scheme
    is analogous to 'sample_1.png', 'sample_2.png', ..., 'sample_n.png'

    Args:
    training_log: Training data (not the full driving log).
    n: Number of random sample images to draw.
    image_size: Size to which cropped input images are resized.
    out_dir: Directory in which to save sample images

    Returns:
    Nothing. Draws images to file.
    """

    image_paths = [path for (path, angle) in sample(training_log, n)]

    for i,path in enumerate(image_paths):
        raw_image = misc.imread(path)

        processed_image = pp.image_preprocess(raw_image, image_size=image_size)
        processed_image = processed_image.reshape(processed_image.shape[:2])

        # The images are drawn side by side.
        figure = plt.figure()
        raw_sp = figure.add_subplot(1, 2, 1)
        raw_sp.set_title("Raw")
        raw_sp.imshow(raw_image)

        pro_sp = figure.add_subplot(1, 2, 2)
        pro_sp.set_title("Processed")
        pro_sp.imshow(cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB))

        output_path = os.path.join(out_dir, 'sample_' + str(i) + '.png')
        figure.savefig(output_path)
