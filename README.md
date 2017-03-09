# Steering Angle Prediction for Simulated Automated Driving

The purpose of this work is personal scholarship - a number of neural network architectures, including pre-trained models (VGG16, InceptionV3, ResNet50) were applied to the problem of steering angle prediction in self driving vehicles. The sources contained here represent the final version of this effort, and include a convolutional network based on a highly efficient architecture published by NVidia. See below for further details.

## Overview
This project contains a very simple 'autopilot' for the Udacity SDCND simulator. Based on visual images alone from a single front-facing camera, a convolutional neural network is applied to predict steering angle, while the vehicle maintains a constant speed (obvious future extensions abound...).

## Data
The training and testing data consists of many thousands of images captured on training laps in the simulator, and the steering angles with which they are associated. Furthermore, many more images are produced by data augmentation, including the use of images from cameras on the sides of the vehicle.

As the training, validation, and testing data is quite large, it is not included in this repository. It is available on request.

## Further details

### Image pre-processing
A very simple pre-processing pipeline is applied to each image.

### Model Architecture
Briefly, the architecture applied here is defined in `model_definition.py` and is implemented with Keras. It is based on the architecture used by NVidia to successfully pilot a vehicle in the publication 'End to End Learnig for Self Driving Cars' **[1]**. On the development machine (see below) the model performs extremely quickly, and training is rapid. The model implemented here is somewhat smaller than that of the original authors, and includes additional facilities for preventing overfitting - deemed necessary due to the limied available training environment. The diagram below illustrates the modified architecture applied here. Of note, the the third 5x5 convolution of the NVidia network is changed to a 3x3 convolution. To save resources, image input to the network are pre-processed more extensively, and are resized to be smaller than those used in the NVidia model.

Note the addition of two dropout layers (with p = 0.5) between the final convolutions, to help mitigate overfitting. When regularization was applied to the fully connected layer, the model tended to become stuck a local minima of the loss function corresponding to a constant steering angle regardless of the input image. After extensive experimentation, only the dropout layers were incorporated into the final model.

<img align='center' src='steernet.png' alt='Steernet Architecture, modified from Bojarski et. al. 2016.'>

Non-negligible image pre-processing is carried out on each image fed to the model - including histogram normalization (CLAHE), cropping, color-space conversion, etc. Further details are to be found in `preprocess.py`.

`training.py` has a simple command line interface. Generally training/testing images are preprocessed separately to speed up model training. The flag `--skip-training` enables the user to halt after pre-processing the training/testing data, without training the model. The flat `--skip-preproc` enables training of the model without re-processing the training/testing data, which is not necessary if simple markdchanges in the model are being evaluated.

## Development Machine
Development was carried out on a MacBook Pro 11'3 (nVidia 750M GPU), running Ubuntu 16.10. Python 3.5.2 and Anaconda 4.2 were used here, and Keras was run with a TensorFlow backend, making full use of the system's GPU.

## Final Thoughts
The network

## References
**[1]** Bojarski, Mariusz, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, et al. 2016. “End to End Learning for Self-Driving Cars.” arXiv [cs.CV]. arXiv. http://arxiv.org/abs/1604.07316.
