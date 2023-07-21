# Emotion-Recognition --- Binary Image Classification

Problem Statement:- We are given a facial image and we need to make a model that classifies whether that image is happy or not happy.

How to build a binary image classifier using convolutional neural network layers in TensorFlow/Keras?
This consists of:
1) Data
2) Model Architecture
3) Test (accuracy)

# Data
We’re going to build an image classifier to classify whether a person is happy or not. I’ve created a small image dataset using my own "smiling" and "not smiling" images taken from the front-facing camera of the smartphone. The data that we fetched earlier is split into the train, test and valid. Then convert train/valid images to the dataset that can be fed to neural network using Keras.preprocessing.image.ImageDataGenerator. This class will automatically label and normalize our data. 

# Model Architecture
At the beginning of this section, we first import TensorFlow. Let’s then add our CNN layers. We’ll first add a convolutional 2D layer with 16 filters, a kernel of 3x3, the input size as our image dimensions, 200x200x3, and the activation as ReLU. After that, we’ll add a max pooling layer that halves the image dimension. We will stack 2 of these layers together, with each subsequent CNN adding more filters. Finally, we’ll flatten the output of the CNN layers, feed it into the first Dense layer, and then to the second Dense layer with a sigmoid function for binary classification. Next, we’ll configure the specifications for model training. We will train our model with the binary_crossentropy loss. We will use the Adam optimizer to reduce the loss function and optimize their weights. We will add accuracy to metrics so that the model will monitor accuracy during training.
