# #!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import csv
import numpy as np
import tensorflow as tf
import sklearn
import random
from keras.models import Model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# #tf.python.control_flow_ops = tf

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

DEV = False
MAX_SHIFT = 30
#Based on visual inspection, for angles of 0.1 the center of the car moves about 60 pixels to the left
PIXELS_PER_ANGLE = 0.15/60 
#Side cameras seems to be 40 to 60 pixels shifted
CORRECTION_FACTOR = 50 * PIXELS_PER_ANGLE
keep_prob = 1.0
MAX_NOISE = 0.05

random.seed(1013)

samples = []
with open("udacity_data/driving_log.csv") as udacity_log:
    reader = csv.reader(udacity_log)
    header = next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
    CENTER, LEFT, RIGHT = (0,1,2)
    ORIENTATION_OPTIONS = [CENTER, LEFT, RIGHT]
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                flip = random.choice([True, False])
                image_orientation = random.choice(ORIENTATION_OPTIONS)
                if image_orientation == CENTER:
                    angle_correction = 0.0
                elif image_orientation == LEFT:
                    angle_correction = CORRECTION_FACTOR
                elif image_orientation == RIGHT:
                    angle_correction = -CORRECTION_FACTOR


                filename = batch_sample[image_orientation].split('/')[-1]
                name = "udacity_data/IMG/" + filename
                train_image = cv2.imread(name)
                noise_factor = random.uniform(1-MAX_NOISE, 1 + MAX_NOISE)
                if float(batch_sample[3]) < 0.001:
                    shift_factor = random.randint(-MAX_SHIFT,MAX_SHIFT)
                    rows,cols = train_image.shape[0:2]
                    M = np.float32([[1,0,shift_factor],[0,1,0]])
                    train_image = cv2.warpAffine(train_image,M,(cols,rows))
                    angle_correction += shift_factor * PIXELS_PER_ANGLE
                center_angle = noise_factor * (float(batch_sample[3]) + angle_correction)
                if not flip:
                    images.append(train_image)
                    angles.append(center_angle)
                else:
                    flip_image = cv2.flip(train_image, flipCode=1)
                    images.append(flip_image)
                    angles.append(-center_angle)

                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


sample_gen = generator(validation_samples, batch_size=1)
sample_instance = next(sample_gen)

print(sample_instance[0].shape)


# Preprocess incoming data, centered around zero with small standard deviation
model = Sequential()
 
model.add(Cropping2D(cropping=((40,10), (MAX_SHIFT,MAX_SHIFT)),
                     input_shape=sample_instance[0].shape[1:]))
model.add(Lambda(lambda x: (x / 128.0) - 1.0))
if DEV:
    model.add(Flatten())
else:
    model.add(Convolution2D(24, 5, 5), 'relu')
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(36, 5, 5), 'relu')
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(48, 5, 5), 'relu')
    model.add(MaxPooling2D((2, 2)), 'relu')
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64, 3, 3), 'relu')
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64, 3, 3), 'relu')
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    model.add(Dense(100), 'relu')
    model.add(Dropout(keep_prob))
    model.add(Dense(50), 'relu')
    model.add(Dropout(keep_prob))
    model.add(Dense(10), 'relu')
    model.add(Dropout(keep_prob))
model.add(Dense(1))
 
model.compile(loss = 'mse', optimizer= 'adam')

if DEV:
    num_epochs = 2
else:
    num_epochs = 10

for i in range(num_epochs):
    model.fit_generator(train_generator, 
                        samples_per_epoch = len(train_samples),  
                        validation_data = validation_generator,
                        nb_val_samples = len(validation_samples),
                        nb_epoch = 1,
                        verbose = 1)


    print("saving model for epoch:", i)
    model.save('model_epoch' + str(i) + '.h5')


# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()



print("Done")
