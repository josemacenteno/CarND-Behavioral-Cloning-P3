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


samples = []
with open("udacity_data/driving_log.csv") as udacity_log:
    reader = csv.reader(udacity_log)
    header = next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                flip = random.choice([True, False])
                filename = batch_sample[0].split('/')[-1]
                name = "udacity_data/IMG/" + filename
                center_image = cv2.imread(name)
                noise_factor = random.uniform(0.96, 1.04)
                center_angle = noise_factor * float(batch_sample[3])
                if not flip:
                    images.append(center_image)
                    angles.append(center_angle)
                else:
                    flip_image = cv2.flip(center_image, flipCode=1)
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
 
model.add(Cropping2D(cropping=((40,10), (0,0)),
                     input_shape=sample_instance[0].shape[1:]))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1))
 
model.compile(loss = 'mse', optimizer= 'adam')
# model.fit(X_train,
#           y_train,
#           validation_split = 0.2,
#           shuffle = True,
#           nb_epoch=2)


history_object = model.fit_generator(train_generator, 
                    samples_per_epoch = len(train_samples),
                    validation_data = validation_generator,
                    nb_val_samples = len(validation_samples),
                    nb_epoch = 5,
                    verbose = 1)


print("saving model")
model.save('model.h5')


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



print("Done")


