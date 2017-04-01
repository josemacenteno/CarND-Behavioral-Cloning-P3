**Behavioral Cloning Project**

The goals / steps of this project are the following:
1. Gather data from a simulator to model good driving behavior
1. Build, a convolution neural network in Keras that predicts steering angles from images
1. Train and validate the model with a training and validation set
1. Test that the model successfully drives around track one without leaving the road
1. Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points ##
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality ###

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode ####

My project includes the following files:
* [model.py](https://github.com/josemacenteno/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/josemacenteno/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [get_model_weights.sh](https://github.com/josemacenteno/CarND-Behavioral-Cloning-P3/blob/master/get_model_weights.sh) A script to download the [model.h5](https://www.dropbox.com/s/oplczho6myqqul1/model.h5) file containing a trained convolution neural network from a [Dropbox link](https://www.dropbox.com/s/oplczho6myqqul1/model.h5)
* [README.md](https://github.com/josemacenteno/CarND-Behavioral-Cloning-P3/blob/master/README.md) summarizing the results
* [video.mp4](https://github.com/josemacenteno/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) is a short video to show how the model performs in the simulator running in Autonomous mode. The lap is recorded using the lowest graphics settings on an intel core i5 system.


#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
 
The model.py file contains the code for training and saving the convolution neural network. The file shows the generator I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a neural network, where the architecture has been copied from a paper published by [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and showcased in a video lecture from Udacity. In short it is a series of 5 Convolutional Neural Network layers followed by 4 fully connected layers and final neuron to produce the angle. 

I selected RELU activation functions on each Convolution2D or Dense layers to introduce nonlinearity (code lines 163-185),  the data is normalized in the model using a Keras lambda layer (code line 160) and cropped with another kearas fucntion Cropping2D (code line 157). 

#### 2. Attempts to reduce overfitting in the model

I included dropout layers in order to reduce overfitting, but the experiments showed that using dropout, even at keep probability of 0.8 hurt the performance in the simulator, so I ended up training without really using the Dropout technique. (model.py line 33)

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 61, 143 and 144). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 189).

I tried a bunch of loss functions I found in the Keras documentation for the model.compile object. Nothing performed better than the mean square error originally sugested by Udacity lectures, so I kept it.  

#### 4. Appropriate training data

Training data was leveraged from a training set provided by Udacity. I planned to add recovery recordings as needed, but the set provided turned out to be sufficient when combined with other data augementation techniques.  The Udacity data set includes midle lane driving laps going backwards mainly. It comes with side camera records too, which was leveraged for data augmentation and calibration. 

For details about how I augmented the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try LeNet and the NVIDIA paper first. The NVIDIA architecture was able to complete a lap on its own, without overfitting after 10 epochs of trainning. LeNet on the other hand showed erratic behavior and failed after the first bridge in the required track. The final submission uses the larger model from the NVIDIA paper, as it proved to have superior performance.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Training on LeNet set a loss metric of 0.5 as a benchmark on how well the model was being optimized. After switching to the NVIDIA model I saw loss go down to 0.03 consistently and sometimes up to 0.013. The car did not complete laps at that stage though.

After augmenting the data set I saw various techniques get the train and validation loss metrics down to 0.0098. The secret sauce was simply to augment the data, specially reducing 0 angle instances.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. 

#### 2. Final Model Architecture

The details of the NVIDIA architecture used can be observed in model,py (code lines 162-187) and in the following table:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:| 
| Convolution | 5x5 patch, 24 filters, ReLU activation function	|
| Max pooling	     | 2x2 stride |
| Convolution | 5x5 patch, 36 filters, ReLU activation function	|
| Max pooling	     | 2x2 stride |
| Convolution | 5x5 patch, 48 filters, ReLU activation function	|
| Max pooling	     | 2x2 stride |
| Convolution | 3x3 patch, 64 filters, ReLU activation function	|
| Convolution | 3x3 patch, 64 filters, ReLU activation function	|
| Flatten() | Converts CNN outputs into a linear array for DNN	|
| Fully connected		| Flat input, 1164 output features			|
| Fully connected		| Flat input, 100 output features			|
| Fully connected		| Flat input, 50 output features			|
| Fully connected		| Flat input, 10 output features			|
| Regression			| 1 linear neuron |

####3. Augmentation of the Training Set & Training Process

To capture good driving behavior, I first used Udacity's data for center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

At this point data is split into train and validation sets. This helps us evaluate train augmentaed data, but keep a validation loss metric based purely on observed behavior.

To augment the data set, I flipped images and angles thinking that this would balance the training between right and left turning.

When looking at thte training data it is clear thet most of the recorded behavior correspongs to a 0.0 degree steering angle. This causes concern for overfitting the model into a car that avoids turning as much as possible. To diversify this training points an image translation that shifts the image horizontally was implemented. Using function "cv2.warpAffine" a trainning image with a very small steering angle is shifted to the right or left at random by 0 to 20 pixels. Careful comparison of larger steering angle images led me to believe that every 60 pixels there needs to be a steering angle between 0.1 and 0.15 towards the center to make the car get back into te center. This 0.15 degress per 60 pixels is used to adjust steering angles for shifted images.

The next obvious augmentation is to use the side camera images recorded. After carefully comparing a center camera image and a side comera image, the side camera is similar to the center image shifted by 40 to 60 pixels. When using a side camera image the angle is adjusted to account for a 50 pixel shift. The camera to be used is uniformily randomized to select between CENTER, LEFT or RIGHT cameras.

Finally a non-intuitive way to augment the data and reduce overfitting is include some noise on the steering angle itself. This technique was has been used by other Udacity students, notably @aflippo proposed it in a Slack discusion. The idea is that given a any frame, there is more than one angle which can be considered as correct behavior. In reality there is a whole range of angles which could be considered a good response to a given frame from the camera. By multiplying the steering angle from the training set by a number between 0.95 and 1.05 we consider any angle within 5% of the training recording to be a good response. This augments the training data by an infinite number of overlapping training points. 

Finally every image is preprocessed by cropping irrelevant sky, car hood and border pixels. This is accomplished by using a Keras function and it is built into the tensor model itself. The data is also normalized to fall in a [-1,1] range and be center around 0 on every given image.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by multiple runs of up to 20 epochs, where the validation loss stops decreasing constantly after epoch 5. Evethough the validation accuracy is not decreasing, I observed some epoch 8 or 9 models performing better, so I left the code to run 10 epochs by default. 
