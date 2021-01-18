# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/architecture.png
[image2]: ./report/summary.png
[image3]: ./report/center.png
[image4]: ./report/triple.png
[image5]: ./report/flipped.png
[image6]: ./report/histogram1.png
[image7]: ./report/histogram2.png
[vid1]: ./report/video.gif





---
### Files

#### 1. Required files to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Functional code
Using the provided [simulator](https://github.com/sbonab/self-driving-car-sim) and the `drive.py` and `model.h5` file in this repo, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

For details about my approach of developing a behavioral cloner for driving the vehicle in the simulator, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The main steps for training a model to drive around the track 
* Using the [simulator](https://github.com/sbonab/self-driving-car-sim) to collect stream of images for a test run around the track.
* Data augmentation
* Preprocessing images
* Selecting the model architecture
* Training the model
* Testing the results in the autonomous mode of the [simulator](https://github.com/sbonab/self-driving-car-sim).

The objective was the vehicle being able to drive autonomously around the track without leaving the road

#### 2. Creation of the Training Set 

To capture the driving behavior, I recorded a single lap on track 1 using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I used the lowest (640x480) screen resolution and the fastest graphics quality for the simulator to avoid unnecessary high-quality images. This in turn will accelerate the training process. 

After the collection process, I had 8036 number of data points. However, as shown in the histogram below, the majority of these data points are correlated to zero steering. 

![alt text][image6]

To help the model steer better, I have randomly discarded 95% of the images with zero steering command. As a result, the following is the updated distribution of the data points

![alt text][image7]

At each instance, the image outputs of the cameras mounted on left/center/right are captured. I have used all three of these output images. For the right camera image, I subtract the steering command by 0.2 for the right image and add it with 0.2 for the left one. Here is an example of image outputs:

![alt text][image4]

To augment the data set, I also flipped images and angles thinking that this would help generalize the driving. For example, here is an image that has then been flipped:

![alt text][image5]


#### 3. Data generator and memory management

I have created the `DataGenerator` class to generate the batches of the images on the fly. This avoids loading large dataset containing the images to the memory to help minimize the memory requirements. 

To do this, first, I create a `data` dictionary with `train` and `validation` as keys. I assume each of these is a list of sample images, represented by dictionaries with keys `path`, `position`, `flipped`, etc. As a result, instead of loading the images, I have created pointers for the images. 

The `DataGenerator` class has different attributes compatible with keras. This class gets `data['train']` and `data['validation']` as inputs and creates batches of images. To use this custom generator, I have finnaly used `fit_generator()` function from keras.

#### 4. Model Architecture

Since the problem is an end-to-end implementation for driving the vehicle around a track, I have used a similar CNN architecture as [DAVE-2](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). An illustration of this architecture is shown in the image below
![alt text][image1]

This architecture has 5 convolutional and 4 fully connected layers. Summary of the model:
![alt text][image2]

I have two layers of preprocessing. First one converts image values to a value between [-1 1]. The second one crops the upper and lower stripes of the image since these areas are representing the extra elements such as trees, rocks, and car's hood.

#### 5. Training Process

I use 20% of the data into a validation set. 
The model contains dropout layers after each fully connected in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 10 as evidenced by trial and error. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 6. Results
Below, you can see the performance of the trained model in driving the vehicle around the first track of the simulator. 
https://www.youtube.com/watch?v=29Dk6D0Ck68&ab_channel=SimInsider
