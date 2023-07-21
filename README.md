# Emotion-Recognition --- Binary Image Classification

Problem Statement:- We are given a facial image and we need to make a model that classifies whether that image is happy or not happy.

How to build a binary image classifier using convolutional neural network layers in TensorFlow/Keras?
This consists of:
1) Data
2) Model Architecture
3) Test (accuracy)

# Data
We’re going to build an image classifier to classify whether a person is happy or not. 
I’ve created a small image dataset using my own "smiling" and "not smiling" images taken from the front-facing camera of the smartphone. 
The data that we fetched earlier is split into the train, test and valid in an 8:1:1 ratio. Then convert train/valid images to the dataset that can be fed to the neural network using Keras.preprocessing.image.ImageDataGenerator. 
This class will automatically label and normalize our data. 
