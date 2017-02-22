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

[project_video_processed]: ./project_videos/project_video_processed.mp4
[test_video_processed]: ./project_videos/test_video_processed.mp4

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
###Write-up / README

####1. Provide a Write-up / README that includes all the rubric points and how you addressed each one.  You can submit your write-up as markdown or PDF. [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template write-up for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This step was approached in two steps: Since the choice of parameters requires a deep insight into the results for each individual step based on various parameter combinations, an [Exploration](exploration.ipynb) jupyter notebook was setup. In addition, resources found on Medium, in particular [Mohankarthik's blog post](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.htp17z4pe), helped a lot to decide on best color space and the whole set of to-be-chosen parameters. It gave a guidance to not brute force all possible combinations but start with a good recommendation. A third colon was quick and dirty implemented tests found in the [test](test.py) file which helped to get started right away with the provided code from the lectures.

The first step was to load the data set and just explore it. The dataset provided contains a number of cars and non-cars images.
* INFO:root:Number of cars: 8792
* INFO:root:Number of notcars: 8968

Two examples of a `vehicle` class and `non-vehicle` class image is provided as follows for a randomly selected index:

|Car / Non-Car at index 1000|Car / Non-Car at index 4000|
|---|---|
|![alt text][car_nocar]|![alt text][carnocar2]|

All sample images are of the same shape:
* Shape of a car (64, 64, 3)
* Shape of a notcar (64, 64, 3)

Its min/max values ranges are:
* Min/Max of a car 0.988235 0.121569
* Min/Max of a notcar 0.839216 0.227451

This is essentially in range [0, 1]. Note that throughout the pipeline it was essential to not mix up different image formats (PNG and JPEG), and different libraries (matplotlib and cv2) as both differ in little details which can be very problematic (different color channel order RGB vs.. BGR, different max/min range [0, 1] vs.. [0, 255])

I then applied on both samples all three possible feature extraction methods:
* Spatial Bin Features (takes the raw image, resizes and lines all values in a vector),
* Hist Color Features (creates a histogram of each channel individually and appends all to one feature vector) and
* HOG Features (analysis gradients and creates a histogram of those for certain color channels).

This provided a feeling of what these feature extraction methods provide. Initially I used the RGB color space but experimented also for most other channels, which is elaborated in the next section.

This is the "Spatial Bin Features" for a car (left) and a non-car (right) for all three color channels (RGB):
![alt text][spatial]

This is the "Hist Color Features" for a car (left) and a non-car (right) for all three color channels (RGB):
![alt text][colorhist]

This is the "HOG Features" for a car (left) and a non-car (right):
![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

First, different color spaces reveal different aspects of the image and highlight different features. In addition, each of the channels within a color space highlights again various notions of features. It is therefore critical to chose the right channel / color space combination for the HOG extraction.

An example revealing differences of each channel of the 'YCrCb' color space is shown in the following plot, where each channel is individually displayed as a grayscaled image (from left to right: `Y`, `Cr`, `Cb`).

![hasv_ycrcb]

Secondly, the HOG feature extraction performance for all channels is evaluated. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as suggested by the lectures for a car (top) and a non-car (bottom) (from left to right: `Y` grayscaled, `Y` HOG, `Cr` HOG, `Cb` HOG):

![alt text][hog_colorspace]

It can be observed that in particular the first Y-channel of the YCrCb color space reveals a great feature set for the cars shape. It can also be observed that the car features differ significantly from the non-car features. Of course, the chosen samples are obviously very different, so that was expected.

Another promising color space is the HSV color space, as we already used that space during the advanced lane finding project. Following plot denotes the same application on the HSV color space (from left to right: `S` graycaled, `H` HOG, `S` HOG, `V` HOG):

![hog_hsv]

Playing around with the HOG parameters revealed that the suggested parameter setup from the lectures was already very effective. The final choice of parameters is listed here:
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

To check the performance, I observed not just the SVC classification error but also the actual performance on some extracted images (directly from the video). The classification error was mostly in the range of 95% to 98.5% for various combinations. However, it was hard to denote based on this error the overall actual performance.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC implemented in the [search_classify](search_classify.py) file within method `train_svc`. The sklearn package already provides a rich API for SVM/SVCs. Training is performed in following steps:
* All training examples (cars and non-cars) are loaded. We do not have to care about memory issues here and can load all images at once, since the dataset is considerable "small" compared to other ML datasets and comprises a total of about 15k samples).
* Samples are randomized to prevent from training a certain dataset order.
* Features for both classes are extracted using the very same parameters. The extraction method is implemented in `extract_features` in the file [lesson_functions](lesson_functions.py). Note that most code from this file is directly taken from the lecture quizzes since it is well tested and implemented and not to complex to not understand.
* Next, the features are normalized using the `StandardScaler` also from the `sklearn` package to avoid certain features to superimpose others.
* Finally the `svc.fit(X_train, y_train)` method is invoked which trains the SVC.
* Eventually the trained SVC and the Scaler are saved to a pickle file for later reuse.

The `extract_features` method first converts the image into the desired color space and essentially calls all three possible feature extractors - if activated. All extracted features are concatenated to a big feature vector which is fed into the training SVC. Those methods are implemented in the [lesson_functions](lesson_functions.py) file:
* `get_hog_features`
* `bin_spatial`
* `color_hist`

Note that once the image is converted to the desired color space, all features are extracted based on this new color space. Improvements would also allow for different color spaces / feature combinations, but was not further investigated here.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search for various window positions at various scales over a certain area of the image. This area was only restricted in the y-axis since no cars are expected in the (roughly) top half of the image. The x-axis was not restricted as it turned out in both extreme cases (left end and right end) cars can appear.
* `y_start_stop = [370, 670]` # Min and max in y to search in slide_window()
* `x_start_stop = [None, None]` # Min and max in x to search in slide_window()

The decision for the size of the sliding windows fell quite late after the whole pipeline including the heatmap approach was running. The overall idea in my used parameter combination is that a big amount of overlapping windows is more promising than a small amount of "exact" windows. That means that the heatmap approach where outliers are dismissed gained more importance. In addition, the heatmap approach subsequently was required to smooth the resulting bounding box to achieve a nice result.

The implementation is found in the [search_classify](search_classify.py) file in the method `run_svc` which takes an image, the trained svc and X_scaler as input pramameters. In this method, first, a set of five window sizes ranging from 60x60 to 140x140 pixels with is setup.

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

After this, the `search_windows` method in invoked which essentially calls the `single_img_features` method for each sliding window. Here, analogously to the feature extraction method, features (spatial, color, HOG) are extracted. The resulting features are fed into the classifier by `prediction = clf.predict(test_features)` which eventually results in a match (a car is detected, add this sliding window to the bboxes array), or not a match (in case we continue with the next sliding window).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on five scales using YCrCb 1-channel HOG (the Y-channel) features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images showing good cases and bad cases.

|Good cases (minor outlier)|Good cases (no outlier)|
|---|---|
|![alt text][good1]|![alt text][good3]|
|![alt text][good2]|![alt text][good4]|

|Bad cases (Trees detected as cars)|Bad cases (Brightness change cause detections)|
|---|---|
|![alt text][bad1]|![alt text][bad2]|

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are OK as long as you are identifying the vehicles most of the time with minimal false positives.)

* Here's a [link to my project video result][project_video_processed]
* Here's a [link to my test video result][test_video_processed]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then threshed that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

The recorded bounding box positions are kept for a certain duration. The current parameters are chosen as such:
* `kRequiredOccurences = 30`
* `kKeepBoxesIterations = 20`

Here the `kKeepBoxesIterations` denotes the number of frames the bounding boxes stays in the array (or more exact a circular buffer). Note that this parameter is chosen to be quite a high number, making detections of cars more reliable but also limiting the possible detection of fast cars which do not stay in a certain area for quite that time. This is definitively a drawback of my solution but worked well with the provided project video.

Here's an example result showing the heatmap from a series of six frames of the video (at timestamps `280`, `790`, `900`, `960`, `1000` and `1260`), the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on all frames:

|All Boxes|Heatmap| Heatmap Clipped|Resulting Box|
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

The starting point of this project was to grab promising code snippets from the lectures and first try to get some basic results. Since I quite quickly ran into issues how to best set the parameters, I chose to explore the methods in a Jupyter Notebook. With help of some Medium articles I slowly got some working configuration. Next I recorded single images from the video (without applying any feature extraction or such) just to have a good test base for the pipeline (not part of this repo). 

Challenging were the choice of good parameters for each of the steps in the pipeline. In particular the feature extraction part including the choice of color space, color channel for HOG, the question whether to include spatial and color histograms was hard and had big influences on the overall performance of the pipeline.

In the sliding window part, the choice of correct sizes for the windows was impactful. Also, detecting outliers via the heatmap approach included a nice tune of parameters, as I not just added a threshold but also introduced a parameter to tune the time over which bounding boxes were kept, was challenging.

The pipeline might fail if cars which travel at higher speed must be detected, since the heatmap approach requires quite a stable position of the detection. I also cannot say how well it will perform on very different videos since the tuning of parameter is probably overfitted to the project video. Also very far objects are hard to detect, since the sliding windows have a lower limit in size. 

Improvements would be a better sliding window choice (i.e. far away object tend to be smaller than closer ones). Also a machine learning approach would be considerable as the tuning of parameters and in particular the combination of those was hard an each influenced the other.

