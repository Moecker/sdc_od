# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_nocar]: ./output_images/car_nocar.png
[spatial]: ./output_images/spatial.png
[colorhist]: ./output_images/colorhist.png
[hog]: ./output_images/hog.png
[hog_colorspace]: ./output_images/hog_colorspace.png
[project_video_processed]: ./project_videos/project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step was approached two fold: Since the choice of parameters requires a deep insight into the behavior based on various combinations, an [exploration.ipynb](Exploration) jupyter notebook was setup. In addition, ressources found on Medium, in particular https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.htp17z4pe, helped a lot to decide on best color space, and the whole set of to-be-chosen parameters. A third colon was quick and dirty implemented tests found in [test.py](test.py) which helped to get started right away with the provided code from the lectures.

The dataset provided contains 
* INFO:root:Number of cars: 8792
* INFO:root:Number of notcars: 8968

An example of a `vehicle` class and `non-vehicle` class image is provided as follows:
![alt text][car_nocar]

The images are of shape 
* Shape of a car (64, 64, 3)
* Shape of a notcar (64, 64, 3)

and its values range from
* Min/Max of a car 0.988235 0.121569
* Min/Max of a notcar 0.839216 0.227451and min/max 
which is essentially [0, 1].

I then applied all three possible feature extraction methods
* Spatial Bin Features,
* Hist Color Features and
* HOG Features
on the test image to get a feeling what these feature extraction methods provide. Initially I used the RGB color space but experimented also for most other channels, which is elaborated in the next section.

This is the "Spatial Bin Features" for a car (left) and a non-car (right)
![alt text][spatial]

This is the "Hist Color Features" for a car (left) and a non-car (right)
![alt text][colorhist]

This is the "HOG Features" for a car (left) and a non-car (right)
![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as suggested by the lectures for a car (above line) and a non-car (bottom line):
![alt text][hog_colorspace]

It can be observed that in particular the first Y-channel of the YCrCb color space reveals a great feature set for the cars shape. It can also be observed that the car features differ significally from the non-car features.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][project_video_processed_final]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

