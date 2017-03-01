# Steering Angle Prediction for Simulated Automated Driving

The purpose of this work is personal scholarship - a number of neural network architectures, including pre-trained models (VGG16, InceptionV3, ResNet50) were applied to the problem of steering angle prediction in self driving vehicles. The sources contained here represent the final version of this effort, and include a convolutional network based on a highly efficient architecture published by NVidia. See below for further details.

## Overview
This project contains a very simple 'autopilot' for the Udacity SDCND simulator. Based on visual images alone from a single front-facing camera, a convolutional neural network is applied to predict steering angle, while the vehicle maintains a constant speed (obvious future extensions abound..).

## Data
The training and testing data consists of many thousands of images captured on training laps in the simulator, and the steering angles with which they are associated. Furthermore, many more images are produced by data augmentation, including the use of images from cameras on the sides of the vehicle.

As the training, validation, and testing data is quite large, it is not included in this repository. It is available on request.

## Further details
Further details related to model architecture, pre-processing, feature extraction etc. will be added to the document `writeup.md`. Briefly, the architecture applied here is defined in `model_definition.py` and is implemented with Keras. It is based on the architecture used by NVidia to successfully pilot a vehicle in the publication 'End to End Learnig for Self Driving Cars' (Bojarski et. al., 2016). On the development machine (see below) the model performs extremely quickly, and training is rapid. The model implemented here is somewhat smaller than that of the original authors, and includes additional facilities for preventing overfitting - deemed necessary due to the limied available training environment.

Non-negligible image pre-processing is carried out on each image fed to the model - including histogram normalization (CLAHE), cropping, color-space conversion, etc. Further details are to be found in `preprocess.py`.

## Development Machine
Development was carried out on a MacBook Pro 11'3 (nVidia 750M GPU), running Ubuntu 16.10. Python 3.5.2 and Anaconda 4.2 were used here, and Keras was run with a TensorFlow backend, making full use of the system's GPU.