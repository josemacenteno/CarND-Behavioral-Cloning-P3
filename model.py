# #!/home/jcenteno/miniconda3/envs/carnd-term1/bin/python
import cv2
import csv
import numpy as np
import tensorflow as tf
import sklearn
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


####################################################################
# Hyperparameters
# 
# How much is it OK to shift an image when applying "Translation data augmentation"
MAX_SHIFT = 20
# Based on visual inspection, for training angles of 0.1 the center of the car moves about 60 pixels to the left
# In an effort to make the car more responsive when it is away from the center, the angle for data augentation
#     was incremented to 0.15
PIXELS_PER_ANGLE = 0.15/60 
#Side cameras seems to be 40 to 60 pixels shifted
CORRECTION_FACTOR = 50 * PIXELS_PER_ANGLE

# This hyperparameter can be used to tune Dropout. It is kept at 1 since we observed the model performs very bad
# when we use dropout. Perhaps the model is already kind of small for the task, so overfitting is not a big concern.
keep_prob = 1.0

# Based on a post in Slack by @aflippo, we use a "noise augmentation" technique to avoid overfitting.
# The idea behing adding some noise to the steering angle is that givena n image, there is more than one
# correct way to react. If you think about it, there is really a range of angles that can be considered
# good behavior for each driving scenario. MAX_NOISE limits from 0.0 to 1.0 how much is it OK to distort the angle
# from the trainning csv log.
MAX_NOISE = 0.05


# Random initialized in a constant seed to make training data sequence repeatable. This helps to make 
# hyper parameter tunning more consistent
random.seed(1013)



####################################################################
# Read the training data provided by Udacity
samples = []
with open("udacity_data/driving_log.csv") as udacity_log:
    reader = csv.reader(udacity_log)
    #discar csv header data
    header = next(reader)

    #Read all the data
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



####################################################################
# Define a generator
# The generator turns image paths into images and augments data without wasting memory
def generator(samples, batch_size=32, augemnt_data = False):
    # Constants used to augment data proportionally
    CENTER, LEFT, RIGHT = (0,1,2)
    if augemnt_data:
        ORIENTATION_OPTIONS = [CENTER, LEFT, RIGHT]
    else:
        ORIENTATION_OPTIONS = [CENTER]

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Loop through data, use all the original instances as starting point
        # for augmented data
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            #Generate a batch of data
            for batch_sample in batch_samples:
                #Flip images half the time
                flip = augemnt_data and random.choice([True, False])

                #Use side camera images to augment the training set
                image_orientation = random.choice(ORIENTATION_OPTIONS)
                if image_orientation == CENTER:
                    angle_correction = 0.0
                elif image_orientation == LEFT:
                    angle_correction = CORRECTION_FACTOR
                elif image_orientation == RIGHT:
                    angle_correction = -CORRECTION_FACTOR

                # Get the image from the path
                filename = batch_sample[image_orientation].split('/')[-1]
                name = "udacity_data/IMG/" + filename
                train_image = cv2.imread(name)
                
                # Calculate how much to distort the steering angle to apply "noise augmentation"
                if augemnt_data:
                    noise_factor = random.uniform(1-MAX_NOISE, 1 + MAX_NOISE)
                else:
                    noise_factor = 1
            
                # Try to diversify center images by applying "Translation augmentation"
                if augemnt_data and float(batch_sample[3]) < 0.001:
                    shift_factor = random.randint(-MAX_SHIFT,MAX_SHIFT)
                    rows,cols = train_image.shape[0:2]
                    M = np.float32([[1,0,shift_factor],[0,1,0]])
                    train_image = cv2.warpAffine(train_image,M,(cols,rows))
                    angle_correction += shift_factor * PIXELS_PER_ANGLE
                
                # Calculate the angle based on the original training csv data
                # with adjustments for the data augmentation techniches
                center_angle = noise_factor * (float(batch_sample[3]) + angle_correction)
                
                # Implement the "image flip" augmetation
                if not flip:
                    images.append(train_image)
                    angles.append(center_angle)
                else:
                    # Flip the image with cv2 and use the opposite angle to test.
                    flip_image = cv2.flip(train_image, flipCode=1)
                    images.append(flip_image)
                    angles.append(-center_angle)

                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

####################################################################
# Define generator functions
#
# with data augmentation for trainning, 
# without data augmentation for validation
train_generator = generator(train_samples, batch_size=128, augemnt_data = True)
validation_generator = generator(validation_samples, batch_size=128, augemnt_data = False)

# Define a generator just to extract shapes from an instance
instance_gen = generator(validation_samples, batch_size=1)
dummy_instance = next(instance_gen)

print(dummy_instance[0].shape)

####################################################################
#Build the model using keras layers and pre-process functions
model = Sequential()
 
# Preprocess incoming data, crop sky, car hood and small borders from let and right
model.add(Cropping2D(cropping=((25,15), (MAX_SHIFT,MAX_SHIFT)),
                     input_shape=dummy_instance[0].shape[1:]))
# Preprocess incoming data, centered around zero with a range -1.0 to 1.0
model.add(Lambda(lambda x: (x / 128.0) - 1.0))

#Add layers based on the architecture from NVIDIA's paper
model.add(Convolution2D(24, 5, 5, activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36, 5, 5, activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 5, 5, activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Flatten())
# Eventhough it is not super clear in te paper if there really is a 1164 neuron dense layer,
# experiments show the car doesn't perform well in the track without it.
model.add(Dense(1164, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(keep_prob))
model.add(Dense(1))
 
model.compile(loss = 'mse', optimizer= 'adam')

####################################################################
# Train the model
# Train and keep track of model loss and validation loss
num_epochs = 10

loss = []
val_loss = []

for i in range(num_epochs):
    history_object = model.fit_generator(train_generator, 
                                         samples_per_epoch = len(train_samples),  
                                         validation_data = validation_generator,
                                         nb_val_samples = len(validation_samples),
                                         nb_epoch = 1,
                                         verbose = 1)

    loss.append(history_object.history['loss'][0])
    val_loss.append(history_object.history['val_loss'][0])
    
    # Quick way to be able to recall models from intermidiate epochs,
    # The code can be enhanced by using Keras built in utilities.
    print("saving model for epoch:", i)
    model.save('model_epoch' + str(i) + '.h5')



####################################################################
# Plot the loss progression though epochs
plt.plot(loss)
plt.plot(val_loss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



print("Done")
