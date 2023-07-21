###################### Import Libraries ######################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator #to generate labels for images. Ex: Images in happy folder in automatically labelled as happy
from tensorflow.keras.preprocessing import image

###################### Intialize 2 classes for training ######################
#Rescale is used to range values from 0 to 1 instead of 0 to 255 (RGB)
train = ImageDataGenerator(rescale=1/255) 
valid = ImageDataGenerator(rescale=1/255)

###################### Convert Train/Valid Images to Dataset that can be feeded to neural network ######################
# flow from directory function is used for labelling
train_dataset = train.flow_from_directory('basedata/train/', 
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')

valid_dataset = valid.flow_from_directory('basedata/valid/',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')

#To see labels and data generated with above functions
train_dataset.class_indices 

###################### Define the Model ######################
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(), 
                                    ###
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    ####
                                    tf.keras.layers.Dense(1,activation='sigmoid')
  
])

###################### Model Compile ######################
model.compile(loss='binary_crossentropy', 
             optimizer='adam',
             metrics='accuracy')

###################### Fit the Model ######################
model_fit = model.fit(train_dataset, 
                     steps_per_epoch=3,
                     epochs=1,
                     validation_data = valid_dataset)

###################### Check the trained model on test data ######################
dir_path = 'basedata/test'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    value = model.predict(images)
