'''
Created on Nov 4, 2017

@author: kiniap
'''
import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

lines = []

'''
Read the various file obtained from training on the simulator

Input: csv files with path to center, left and righ images, and the steering angle command
Output: All the lines from the various csv files
'''
# Original data provided was NOT used in the final model
# with open('./data/driving_log0.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)# skip header
#     for line in reader:
#         lines.append(line)

'''
Process data collected while driving CCW on trk1: run1
''' 
with open('./data/driving_log_trk1_ccw_r1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)        

'''
Process data collected while driving CCW on trk1: run2
''' 
with open('./data/driving_log_trk1_ccw_r2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) 

'''
Process data collected while driving CW on trk1: run1
'''         
with open('./data/driving_log_trk1_cw_r1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

'''
Define the various parameters of the network
'''

BATCH_SIZE = 128 # batch size of 128 for the generator. Seems to work better than 32
CORRECTION = 0.25 # Steering correction applied to get the left and right steering angles (only center is provided)
EPOCHS = 3 # Number of epochs to run the training for

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("Number of training samples: ", len(train_samples))
print("Number of validation samples: ", len(validation_samples))

'''
Generator function: Extract the camera images and the corresponding steering angles
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            camera_images = []
            steering_angles = []

            for batch_sample in batch_samples:
                
                '''
                Process the center image and the corresponding steering angle command
                '''
                file_path = batch_sample[0]  # Extract the center image
                filename = file_path.split('/')[-1]  #  Extract filename
                center_image = cv2.imread('./data/IMG/'+filename)  # read in the image
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)  # covert to RGB, by default cv2.imread is BGR
                center_image_flipped = np.fliplr(center_image) # flip the center image, to create mirror image data
                
                center_steering = float(line[3]) # Extract the steering angle
                center_steering_flipped = -center_steering #Flipped steering angle
                '''
                Randomly skip if the steering angle is close to zero wth 60% probability
                This is to avoid overfitting around zero steering angle
                '''
#               prob = np.random.random()   # generates a random number between 0 and 1
#               if (prob < 0.6) and (abs(center_steering) < 0.75):
#                   continue;                
                '''
                Add center_images and steering angles to the list
                '''
                camera_images.append(center_image)
                camera_images.append(center_image_flipped)
                steering_angles.append(center_steering)  
                steering_angles.append(center_steering_flipped)    
    
                '''
                Process the left image and add a correction to the center steering angle: center steering + correction
                '''
                file_path = batch_sample[1]
                filename = file_path.split('/')[-1]
                left_image = cv2.imread('./data/IMG/'+filename)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) # convert left image to RGB
                left_image_flipped = np.fliplr(left_image)  # Flipped left image
                  
                left_steering = center_steering+CORRECTION  
                left_steering_flipped = -left_steering
                
                '''
                Add left_images and the steering angles to the list
                '''                
                camera_images.append(left_image)
                camera_images.append(left_image_flipped)
                steering_angles.append(left_steering)
                steering_angles.append(left_steering_flipped) 
    
                '''
                Process the right image and add a correction to the center steering angle: center steering - correction
                '''
                file_path = batch_sample[2]
                filename = file_path.split('/')[-1]
                right_image = cv2.imread('./data/IMG/'+filename)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) # convert right image to RGB
                right_image_flipped = np.fliplr(right_image) # Flipped right image  
                
                right_steering = center_steering-CORRECTION
                right_steering_flipped = -right_steering  
                  
                '''
                Add right_images and the steering angles to the list
                '''  
                camera_images.append(right_image)
                camera_images.append(right_image_flipped)
                steering_angles.append(right_steering)
                steering_angles.append(right_steering_flipped) 
            
            # Make sure we atleast have a few images to train on!
            if (len(camera_images) == 0):
                print("\n No camera images in this batch!")
                continue;
                
            # Convert to Numpy arrays and shuffle
            x_train = np.array(camera_images)
            y_train = np.array(steering_angles) 
            y_train = y_train.reshape(-1,1) 
            yield shuffle(x_train, y_train)

'''
Import the various keras utilities to create and train the model
'''
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

'''
Define the training and validation generator functions
'''
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

'''
Preprocess the image data
'''
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 160))))
model.add(Lambda(lambda x: (x/255.0)-0.5))

'''
Implement the Lenet architecture : Did not use this for the final model
'''
# model.add(Convolution2D(6, (5,5), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, (5,5), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

''' 
Implement the Nvidia architecture
'''
model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Convolution2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print(model.summary())

'''
Compile the model with Adam optimizer and use Mean Squared Error as the loss metric
Use a generator to feed data into the model
 
'''
model.compile(loss='mse', optimizer='adam') 
history_object = model.fit_generator(train_generator, 
                    steps_per_epoch = (len(train_samples)*6)//BATCH_SIZE, 
                    validation_data=validation_generator, 
                    validation_steps=(len(validation_samples)*6)//BATCH_SIZE,
                    epochs=EPOCHS)
 
model.save('model_new.h5')
 
### print the keys contained in the history object
print(history_object.history.keys())
  
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

    