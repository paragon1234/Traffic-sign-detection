# **Traffic Sign Recognition using Convolutional Neural Network** 

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./InputDataHistogram.jpg



## Design And Architecture

You're reading it! and here is a link to my [project code](https://github.com/paragon1234/Traffic-signal-detection)

### Data Set Summary & Exploration

#### 1. Basic Features

* The size of training set is 34799 rbg images
* The size of the validation set is 4410 rgb images
* The size of test set is 12630 rgb images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43 


#### 2. Dataset Exploration

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution of classes in the training, validation and test set.

![alt text][image1]


From the above diagram it is clear that the distribution of classes in the training, validation and test set is same. However, the dataset in imbalanced as all the classes do not have same number of samples.  The images also differ significantly in terms of clarity, contrast and brightness. So we will need to apply some kind of histogram equalization, this should noticeably improve feature extraction. There are also rotation, scale and other geometric transformation.



### Design and Test a Model Architecture


#### 1. Pre-processing

1) As a first step, I decided to convert the images to grayscale because the classification feature are independent of the color.
2) Histogram equalizer was applied because the method is useful in images with backgrounds and foregrounds that are both bright or both dark. This is useful when detecting images under dark or bright light. We used adaptive histogram equalization as this particularly useful when the image contains regions that are significantly lighter or darker than most of the image.
3) Finally we converted the gray pixels of the image in the range -0.5 to 0.5

 
#### 2.  model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GraySale image   						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x30	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x30 					|
| Flatten				| outputs 750 									|
| Fully connected		| Output 300        							|
| RELU					|												|
| Fully connected		| Output 43 									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
I have experimented with variation of the above architecture, but either the accuracy was less than this, or equal to it with more computation in each iteration. Hence, I finalized with the above architecture as it gave best performance with least computation. Following variations were tried:
*Increasing depth of covolution layers: conv1 to 32 and conv2 to 64
*Increasing depth of layers: 3 convolution layers and 3 fully connected layers
*Fully connected layer using concatenated data from all the 3 covolution layers
*Adding droup-out of 0.6/0.8 to the fully connected layer
*Adding droup-out of 0.9, 0.8, 0.7 to the 3 convolution layers and a droupout of 0.9 to all the 3 fully connected alyers


#### 3. Architecture Specification
*Adam Optimizer
*rate = 0.001
*EPOCHS = 12
*BATCH_SIZE = 128
*Softmax Cross Entropy with One-Hot Labels


#### 4. Result

My final model results were:
* validation set accuracy of 96.7
* test set accuracy of 95.1
 


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
