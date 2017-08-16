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
[image2]: ./images/doubleCurve.png
[image3]: ./images/yield.png
[image4]: ./images/keepRight.pngng
[image5]: ./images/speedLimit60.png
[image6]: ./images/stop.png


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
* Increasing depth of covolution layers: conv1 to 32 and conv2 to 64
* Increasing depth of layers: 3 convolution layers and 3 fully connected layers
* Fully connected layer using concatenated data from all the 3 covolution layers
* Adding droup-out of 0.6/0.8 to the fully connected layer
* Adding droup-out of 0.9, 0.8, 0.7 to the 3 convolution layers and a droupout of 0.5 to all the 3 fully connected alyers


#### 3. Architecture Specification
* Adam Optimizer
* rate = 0.001
* EPOCHS = 12
* BATCH_SIZE = 128
* Softmax Cross Entropy with One-Hot Labels


#### 4. Result

My final model results were:
* validation set accuracy of 97.5
* test set accuracy of 95.5
 


### Test a Model on New Images

#### 1. Five German traffic signs found on the web
Here are five German traffic signs that I found on the web. Each image is re-sized to 32x32 using irfanView software. The top five probabilities of each image along with the labels are:

![alt text][image2] 
[  9.99988437e-01,   6.16074567e-06,   5.09703432e-06, 1.15665337e-07,   1.07816064e-07]
[21, 25, 12, 11, 23]
Conclusion: Correct with high confidence

![alt text][image3] 
[  1.00000000e+00,   2.44773955e-14,   6.03770040e-15, 3.64478913e-19,   1.46541585e-19]
[13, 12, 35, 38, 15]
Conclusion: Correct with high confidence

![alt text][image4] 
[  1.00000000e+00,   4.10162840e-16,   7.20662497e-18, 6.51623398e-18,   4.57199351e-18]
[38, 34, 36, 25, 40
Conclusion: Correct with high confidence

![alt text][image5] 
[  9.55771565e-01,   4.15060930e-02,   1.27782230e-03, 6.56668795e-04,   3.70674330e-04]
[40, 35, 37, 38, 12]
Conclusion: Wrong with high probability. The image is not clear because of snow. It is predicting "Round About Mandatory" because of circular shape. The correct label is not within top 5.

![alt text][image6]
[  9.99993682e-01,   6.35076822e-06,   4.25740154e-08, 2.45831959e-08,   1.68262364e-08]
[14, 17, 38, 34,  9]
Conclusion: Correct with high confidence. This image has a very dark contrast. However, as we have applied adaptive histogram equalizer, we are able to predict it correctly.


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.

