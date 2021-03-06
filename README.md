#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[left]: ./imgs/left.jpg "Left camera"
[left_flip]: ./imgs/left_flip.jpg "Left camera flip"
[center]: ./imgs/center.jpg "Center camera"
[center_flip]: ./imgs/center_flip.jpg "Center camera flip"
[right]: ./imgs/right.jpg "Right camera"
[right_flip]: ./imgs/right_flip.jpg "Right camera flip"
[loss_graph]: ./imgs/loss_graph.png "Loss graph"
[angle_hist]: ./imgs/angle_histogram.png "Angle histogram"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Here, I chose to use the model for autonomouse driving that was used by NVidia. At first, three convolutional layers are used, each having a 5x5 filter size, and then passed on to two different convolutional layers with a 3x3 filter size each. The data is then flattened, and sent through three fully connected layers before producing an output.

For improving performance, a pre-processing functions was written and called before using the NVidia model mentioned above. First, the data is normalized with zero mean, then cropped since all the data from each image is not necessary, and finally resized to a smaller format to lessen computation time during training.

#### Attempts to reduce overfitting in the model
While training the network, the data is first shuffled before being fed into training using a generator. Apart from that, a dropout layer was used after the convolutional layers and before the data is flattened.

Some data augmentation has also been done in the form of flipping the images horizontally, and using the cameras. Horizontally flipping the images allows the vehicle to not overfit the curves, and using the side cameras helps with recovery driving if the car gets close to the sides.

Below, three sets of images can be side by side, displaying how each camera's image was flipped.

Left camera:
![alt text][left]  ![alt text][left_flip] 

Center camera:
![alt text][center] ![alt text][center_flip] 

Right camera: 
![alt text][right] ![alt text][right_flip]


Traditionally, the training and validation loss are also used to reduce overfitting. This is done by checking how the training and validation loss change after each epoch. The model is considered to be overfitted if the validation loss ceases to change between each epoch, or even increases, while the training loss decreases. Below, you will find a plot of how the training and validation loss change over epochs, using a train/validation split of 80/20 over a total of around 41000 images.

![alt text][loss_graph] 

The graph displays the change in loss over the course of 20 EPOCHS. In our case for the driving car, this method proved to undependable, since the training and validation loss were a poor measure of the vehicle's performance to begin with. The resluting video of the self-driving car was only produced using two epochs, since more epochs did not display any signs of large increase in performance.


#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. A significant amount of data was tested and discarded, but what I found gave me the best results was using three laps of center lane driving, and two laps of center lane driving in reverse. This combined with the data augmentation proved sufficient. However, the vehicle does not drive as smoothely as a human would, so something that could be of benefit here would be to add a lap of smooth driving, and adding some driving data around specific areas that may not completely satisfy the vehicle's performance.


A histogram comparison between the provided data from udacity and my data can be seen below.

![alt text][angle_hist]

Here it can clearly be seen that Udacity's data contains a somewhat larger bias towards zero angle, and a less even distribution towards the other angles. By using the mouse while steering during recordings from the simulations, a good distribution can be obtained, since the mouse usually leads to a small angle, making the model less biased towrads having an angle of zero.


### Model Architecture and Training Strategy


First, the LeNet model architecture was tested for autonomous driving. While this network was able to drive the vehicle, it was found to be slightly lacking. Searching for a more complex architecture, NVidia's model proved to do exactly what this project requires. Namely to provide a steering angle based on a vehicle's position on the road. Below, the arcitecture of the model can be seen, which was generated by the model.summary() command provided by Keras.

Layer (type)                 Output Shape              Param #   	Comment
===================================================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         	Nomralizing with 0 mean
___________________________________________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         	Image cropping
___________________________________________________________________________________________________
lambda_2 (Lambda)            (None, 64, 64, 3)         0      		Image resizing
___________________________________________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 24)        1824      	Convolution
___________________________________________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 36)        21636     	Convolution
___________________________________________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 5, 48)          43248     	Convolution
___________________________________________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 64)          27712     	Convolution
___________________________________________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 64)          36928     	Convolution
___________________________________________________________________________________________________
dropout_1 (Dropout)          (None, 1, 1, 64)          0         	Dropout
___________________________________________________________________________________________________
flatten_1 (Flatten)          (None, 64)                0         	Flatten
___________________________________________________________________________________________________
dense_1 (Dense)              (None, 100)               6500      	Fully connected layer
___________________________________________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      	Fully connected layer
___________________________________________________________________________________________________
dense_3 (Dense)              (None, 10)                510       	Fully connected layer
___________________________________________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        	One output
===================================================================================================


The data that was collected was split into a training and validation set, where the training data contained 80% of the samples taken, and the validation set contained 20%. The error was calculated using the Mean Square Error (MSE), but it was found that the validation error was not an accurate measure of how well a model performed.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
