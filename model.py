#!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import csv
import numpy as np
import tensorflow as tf

#tf.python.control_flow_ops = tf

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []

with open("udacity_data/driving_log.csv") as udacity_log:
    reader = csv.reader(udacity_log)
    header = next(reader)
    for line in reader:
        lines.append(line)


def get_next_data_point():
    for line in lines:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        image_path = "udacity_data/IMG/" + filename
        image = cv2.imread(image_path)
        measured_angle = float(line[3])
        yield image, measured_angle

#Build list
images = []
measured_angles = []
index = 0
for image, measured_angle in get_next_data_point():
    images.append(image)
    measured_angles.append(measured_angle)

    flip_image = cv2.flip(image, flipCode=1)
    images.append(flip_image)
    measured_angles.append(-measured_angle)

    # if index == 100:


X_train = np.array(images)
y_train = np.array(measured_angles)

print(X_train.shape, y_train.shape)

model = Sequential()

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=X_train.shape[1:]))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer= 'adam')
model.fit(X_train,
          y_train,
          validation_split = 0.2,
          shuffle = True,
          nb_epoch=2)

print("saving model")
model.save('model.h5')

print("Done")