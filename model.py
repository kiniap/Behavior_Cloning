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

lines = []

'''
Read the driving_log.csv file
'''
# with open('./data/driving_log0.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)# skip header
#     for line in reader:
#         lines.append(line)

with open('./data/driving_log_trk1_ccw_r1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)        

with open('./data/driving_log_trk1_ccw_r2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line) 
        
with open('./data/driving_log_trk1_cw_r1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)   
#print(lines[0][0])


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print("Number of training samples: ", len(train_samples))
print("Number of validation samples: ", len(validation_samples))


BATCH_SIZE = 128
'''
Generator function: Extract the camera images and the corresponding steering angles
'''
def generator(samples, batch_size=32, validation_pass = False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            camera_images = []
            steering_angles = []
            correction = 0.25
            for batch_sample in batch_samples:
                # Extract the center image
                file_path = batch_sample[0]
                filename = file_path.split('/')[-1]
                center_image = cv2.imread('./data/IMG/'+filename)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # Extract the steering angle
                center_steering = float(line[3])
                
                # randomly skip if  steering angle is close to zero..avoid overfitting around zero
#                 prob = np.random.random()   # generates a random number between 0 and 1
#                 if (prob < 0.6) and (abs(center_steering) < 0.75) and (validation_pass == False):
#                     continue;
#                 
                # Append center_image, steering angle
                camera_images.append(center_image)
                steering_angles.append(center_steering)   
    
                # Flipped center image
                center_image_flipped = np.fliplr(center_image)
                center_steering_flipped = -center_steering
                camera_images.append(center_image_flipped)
                steering_angles.append(center_steering_flipped)    
    
                # Extract the left image
                file_path = batch_sample[1]
                filename = file_path.split('/')[-1]
                left_image = cv2.imread('./data/IMG/'+filename)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) # convert left image to RGB
                camera_images.append(left_image)
                left_steering = center_steering+correction
                steering_angles.append(left_steering)
                
                # Flipped left image
                left_image_flipped = np.fliplr(left_image)
                left_steering_flipped = -left_steering
                camera_images.append(left_image_flipped)
                steering_angles.append(left_steering_flipped) 
    
                # Extract the right image
                file_path = batch_sample[2]
                filename = file_path.split('/')[-1]
                right_image = cv2.imread('./data/IMG/'+filename)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) # convert right image to RGB
                camera_images.append(right_image)
                right_steering = center_steering-correction
                steering_angles.append(right_steering)
                
                # Flipped right image
                right_image_flipped = np.fliplr(right_image)
                right_steering_flipped = -right_steering
                camera_images.append(right_image_flipped)
                steering_angles.append(right_steering_flipped) 
            
            if (len(camera_images) == 0):
                print("\n No camera images in this batch!")
                continue;
                
            # trim image to only see section with road
            x_train = np.array(camera_images)
            y_train = np.array(steering_angles) 
            y_train = y_train.reshape(-1,1) 
            yield shuffle(x_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

'''
Create and train the model
'''
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


#print("Shape of features: ", x_train.shape)  
#print("Shape of labels: ", y_train.shape)

# Preprocess the image data
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 160))))
model.add(Lambda(lambda x: (x/255.0)-0.5))

# Implement the Lenet architecture
# model.add(Convolution2D(6, (5,5), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, (5,5), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# Implement the Nvidia architecture
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

model.compile(loss='mse', optimizer='adam') 
history_object = model.fit_generator(train_generator, 
                    steps_per_epoch = (len(train_samples)*6)//BATCH_SIZE, 
                    validation_data=validation_generator, 
                    validation_steps=(len(validation_samples)*6)//BATCH_SIZE,
                    epochs=3)

model.save('model.h5')

# ### print the keys contained in the history object
# print(history_object.history.keys())
# 
# import matplotlib
# matplotlib.use('agg') 
# import matplotlib.pyplot as plt
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

    