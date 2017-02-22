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
[carnocar2]: ./output_images/carnocar2.png

[spatial]: ./output_images/spatial.png
[colorhist]: ./output_images/colorhist.png
[hog]: ./output_images/hog.png
[hog_colorspace]: ./output_images/hog_colorspace.png
[hog_hsv]: ./output_images/hog_hsv.png
[hasv_ycrcb]: ./output_images/hasv_ycrcb.png
[project_video_processed_final]: ./project_videos/project_video_processed_final.mp4

[good1]: ./output_images/good_img_260_processed.png
[good2]: ./output_images/good_img_620_processed.png
[good3]: ./output_images/good_img_840_processed.png
[good4]: ./output_images/good_img_960_processed.png

[bad1]: ./output_images/bad_img_440_processed.png
[bad2]: ./output_images/bad_img_540_processed.png

[chain280a]: ./output_images/heat_all_img_280_processed.png
[chain280b]: ./output_images/heat_map_img_280_processed.png
[chain280c]: ./output_images/heat_clip_gray_img_280_processed.png
[chain280d]: ./output_images/heat_resultimg_280_processed.png

[chain780a]: ./output_images/heat_all_img_780_processed.png
[chain780b]: ./output_images/heat_map_img_780_processed.png
[chain780c]: ./output_images/heat_clip_gray_img_780_processed.png
[chain780d]: ./output_images/heat_resultimg_780_processed.png

[chain900a]: ./output_images/heat_all_img_900_processed.png
[chain900b]: ./output_images/heat_map_img_900_processed.png
[chain900c]: ./output_images/heat_clip_gray_img_900_processed.png
[chain900d]: ./output_images/heat_resultimg_900_processed.png

[chain960a]: ./output_images/heat_all_img_960_processed.png
[chain960b]: ./output_images/heat_map_img_960_processed.png
[chain960c]: ./output_images/heat_clip_gray_img_960_processed.png
[chain960d]: ./output_images/heat_resultimg_960_processed.png

[chain1000a]: ./output_images/heat_all_img_1000_processed.png
[chain1000b]: ./output_images/heat_map_img_1000_processed.png
[chain1000c]: ./output_images/heat_clip_gray_img_1000_processed.png
[chain1000d]: ./output_images/heat_resultimg_1000_processed.png

[chain1260a]: ./output_images/heat_all_img_1260_processed.png
[chain1260b]: ./output_images/heat_map_img_1260_processed.png
[chain1260c]: ./output_images/heat_clip_gray_img_1260_processed.png
[chain1260d]: ./output_images/heat_resultimg_1260_processed.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step was approached in two steps: Since the choice of parameters requires a deep insight into the results for each individual step based on various paramater combinations, an [Exploration](exploration.ipynb) jupyter notebook was setup. In addition, ressources found on Medium, in particular [Mohankarthik's blog post](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.htp17z4pe), helped a lot to decide on best color space and the whole set of to-be-chosen parameters. It gave a guidance to not brute force all possible combinations but start with a good recommendaion. A third colon was quick and dirty implemented tests found in the [test](test.py) file which helped to get started right away with the provided code from the lectures.

the first step was to load the data set and just explore it.

The dataset provided contains 
* INFO:root:Number of cars: 8792
* INFO:root:Number of notcars: 8968

An example of a `vehicle` class and `non-vehicle` class image is provided as follows:

|Example index 1000|Example index 4000| 
|---|---|
|![alt text][car_nocar]|![alt text][carnocar2]|

The images are of shape 
* Shape of a car (64, 64, 3)
* Shape of a notcar (64, 64, 3)

and its values range from
* Min/Max of a car 0.988235 0.121569
* Min/Max of a notcar 0.839216 0.227451

which is essentially [0, 1]. Note that throughout the pipeline it was essential to not mix up different image formats (PNG and JPEG), and different libraries (matplotlib and cv2) as both have little details which can be very problematic (different color channel order, different max/min range)

I then applied on bith examples all three possible feature extraction methods
* Spatial Bin Features,
* Hist Color Features and
* HOG Features

to get a feeling what these feature extraction methods provide. Initially I used the RGB color space but experimented also for most other channels, which is elaborated in the next section.

This is the "Spatial Bin Features" for a car (left) and a non-car (right) for all three color channels on a simple plot:
![alt text][spatial]

This is the "Hist Color Features" for a car (left) and a non-car (right) for all three color channels on a simple plot:
![alt text][colorhist]

This is the "HOG Features" for a car (left) and a non-car (right)
![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

First, different color spaces reveal different aspects of the image and highlight different features. In addition, each of the channels within a color space highlights again various notions of features. It is therefore critical to chose the right channel / color space combination for the HOG extraction. 

An example of the three channels of the 'YCrCb' color space is shown in the following plot, where each channel is individually displayed as a grayscaled image.

![hasv_ycrcb]

Secondly, the HOG feature extraction performance for all channels is evaluated. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as suggested by the lectures for a car (top) and a non-car (bottom):

![alt text][hog_colorspace]

It can be observed that in particular the first Y-channel of the YCrCb color space reveals a great feature set for the cars shape. It can also be observed that the car features differ significally from the non-car features.

Another promising color space is the HSV color space, as we already used that space during the advanced lane finding project. Following the same application on the HSV color space using the Saturation channel `S` for HOG feature extraction.

![hog_hsv]

Playing around with the HOG parameters revealt that the suggetsed parameter setup of 
* `orientations=9` # Number of HOG extracted orientations
* `pixels_per_cell=(8, 8)` Number of pixels per HOG cell
* `cells_per_block=(2, 2)` Number of cells per HOG block
* `color_space = 'YCrCb'` # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* `hog_channel = 0` # Can be 0, 1, 2, or "ALL"
* `spatial_size = (32, 32)` # Spatial binning dimensions
* `hist_bins = 64` # Number of histogram bins
* `spatial_feat = True` # Spatial features on or off
* `hist_feat = True` # Histogram features on or off
* `hog_feat = True` # HOG features on or off

was already most effective. To check the performance not just the SVC classification error but the actual performance on some extracted images (directly from the video) were most helpful.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC implemented in the `search_classify.py` file within method `train_svc`. The sklearn package already provides a rich API for SVM/SVCs. Training is performed using following steps:
* All training examples (for cars and non-cars) are loaded. We do not have to care about memory issues since the dataset is coniderable "small" compared to other ML datatsets of about 15k samples)
* Samples are randomized
* Features for both classes are extracted using the very same parameters. The extraction method is implemented in `extract_features` in the file `lesson_functions.py`. Note that most code form this file is taken directly from the lecture quizzes since it is well tested and implemented and not to complex to not understand.
* Next, the features are normalized using the StandardScaler also from the sklearn package to avoid certain features to superimpose others.
* Finally the `svc.fit(X_train, y_train)` method is invoked which trains the SVC. 
* Eventually the trained SVC and the Scalar are saved to a pickle file for later reuse.

The `extract_features` method first converted the image into the desired color space and basically called all three possible feature extractors - if activated. All extracted features are concatenated to a big feature vector which is fed into the training.
* `get_hog_features`
* `bin_spatial`
* `color_hist`

Note that once the image is converted all features are extracted based on this new color space. Improvemtents would also allow for different color space - feature combinations, but was not further investigated here.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search for various window positions at various scales over a certain area of the image. This are was only restricted in the y-axis since no cars are expected in the top part of the image. The x-axis was not restricted as it turned out in both extreme cases (left and right) cars can appear.
* `y_start_stop = [370, 670]` # Min and max in y to search in slide_window()
* `x_start_stop = [None, None]` # Min and max in x to search in slide_window()

The decision for the size of the sliding windows fell quite late after the whole pipeline including the heatmap approach was running. The overall idea in my used parameter combination is that rather a big amount of overlapping windows is more promising than small amount of "exact" windows. That means that the heatmap approach where outliers are dismissed gained more importance. In additon, the heatmap approach subsequentially was reuquired to smooth the resulting bounding box to a nice result.

The implementation is found in the [search_classify](search_classify.py) file in the method `run_svc` which takes an image, the trained svc and X_scaler as input params. First a set of five window sizes ranging from 60x60 to 140x140 pixels is setup.

```
windows = []
for size in range(60, 140, 20):
    wins = slide_window(image,
                        x_start_stop=x_start_stop,
                        y_start_stop=y_start_stop,
                        xy_window=(size, size),
                        xy_overlap=(0.5, 0.5))  
    windows = windows + wins    
```

After this, the `search_windows` method in invoked which essentially calls the `single_img_features` method for each sliding window. Here, analogously to the feature extraction method, features (sptial, color, HOG) are extracted. The resulting features are fed into the classifier by `prediction = clf.predict(test_features)` which eventually results in a match (a car is detected, add this sliding window to the bboxes array), or not a match (in case we continue with the next sliding window)

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 1-channel HOG )(the Y-channel) features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images showing good cases and bad cases.

|Good cases (minor outlier)|Good cases (no outlier)| 
|---|---|
|![alt text][good1]|![alt text][good3]|
|![alt text][good2]|![alt text][good4]|

|Bad cases (Trees detected as cars)|Bad cases (Brigthnes change cause detections)| 
|---|---|
|![alt text][bad1]|![alt text][bad2]|

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result][project_video_processed_final]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

The recorded bounding box positions are kept for a certain duration. The current parameters are chosen as such:
* `kRequiredOccurences = 30`
* `kKeepBoxesIterations = 20`

where the `kKeepBoxesIterations` denotes the number of frames the bounding boxen stay in the array. Note that this parameter is quite high, making detections of cars more reliable but also limits the possible detection of fast cars which do not stay in a certain area for quite a time. This is defenitevly a drawback of my solution but worked well with the provided project video.

Here's an example result showing the heatmap from a series of six frames of the video (at timestmaps 280, 790, 900, 960, 1000 and 1260), the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on all frames:

|All Boxes|Heatmap| Heatmap Cliped|Resulting Box|
|---|---|---|---|
|![chain280a]|![chain280b]|![chain280c]|![chain280d]|
|![chain780a]|![chain780b]|![chain780c]|![chain780d]|
|![chain900a]|![chain900b]|![chain900c]|![chain900d]|
|![chain960a]|![chain960b]|![chain960c]|![chain960d]|
|![chain1000a]|![chain1000b]|![chain1000c]|![chain1000d]|
|![chain1260a]|![chain1260b]|![chain1260c]|![chain1260d]|

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Challenging were the choice of good parameters for each of the steps in the pipeline. In particular the feature extraction part including the choice of color space, color channel for HOG, the question whether to include patial and color histograms was hard and had big influences on the overall performance of the pipeline. 

In the sliding window part, the choice of correct sizes for the windows was impactful. Also detecting outliers via the heatmap approach included a nice tune of parameters, as I not just added a threshold but also introduced a parameter to tune the time over which bounding boxes were kept.

