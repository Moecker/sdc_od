# This file holds methods to classify images, train the SVC
# as well as implements the sliding window approach

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob
import time
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from lesson_functions import *

# The parameters used for the feature extraction methods
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

# Parameters used for the sliding window methods
y_start_stop = [370, 670] # Min and max in y to search in slide_window()
x_start_stop = [None, None] # Min and max in y to search in slide_window()
sample_size = 8700


# Runs the svc chain on the given image with fiven svc classifier and scale
def run_svc(image, svc, X_scaler):
    draw_image = np.copy(image)
    
    # Here we create five sliding window sizes
    windows = []
    for size in range(60, 140, 20):
        wins = slide_window(image,
                            x_start_stop=x_start_stop,
                            y_start_stop=y_start_stop,
                            xy_window=(size, size),
                            xy_overlap=(0.5, 0.5))  
        windows = windows + wins                
      
    # Search for "hot" windows where a car could be located
    hot_windows = search_windows(image,
                                 windows, 
                                 svc, 
                                 X_scaler, 
                                 color_space=color_space, 
                                 spatial_size=spatial_size, 
                                 hist_bins=hist_bins, 
                                 orient=orient, 
                                 pix_per_cell=pix_per_cell, 
                                 cell_per_block=cell_per_block, 
                                 hog_channel=hog_channel, 
                                 spatial_feat=spatial_feat, 
                                 hist_feat=hist_feat, 
                                 hog_feat=hog_feat, 
                                 hist_range=(0, 1))                       

    window_img = draw_boxes(draw_image, 
                            hot_windows, 
                            color=(0, 0, 255),
                            thick=6)      
                            
    return window_img, hot_windows


# Defines a function that takes an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, 
                   windows,
                   clf,
                   scaler,
                   color_space, 
                   spatial_size,
                   hist_bins,
                   orient, 
                   pix_per_cell,
                   cell_per_block, 
                   hog_channel,
                   spatial_feat, 
                   hist_feat,
                   hog_feat,
                   hist_range):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient, 
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img,
                        color_space,
                        spatial_size,
                        hist_bins,
                        orient, 
                        pix_per_cell,
                        cell_per_block,
                        hog_channel,
                        spatial_feat,
                        hist_feat,
                        hog_feat):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=(0, 1))
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient,
                                    pix_per_cell,
                                    cell_per_block, 
                                    vis=False,
                                    feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel],
                                            orient, 
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

   
def train_svc():    
    # Read in cars and notcars
    images_car = glob.glob('data/vehicles/*/*.png')
    images_notcar = glob.glob('data/non-vehicles/*/*.png')
    cars = []
    notcars = []
       
    for image in images_car:
        cars.append(image)
        
    for image in images_notcar:
        notcars.append(image)

    print("Num cars: " + str(len(cars)))
    print("Num not-cars: " + str(len(notcars)))
    
    import random
    cars = random.sample(cars, sample_size)
    notcars = random.sample(notcars, sample_size)
    
    print("Train num cars: " + str(len(cars)))
    print("Train num not-cars: " + str(len(notcars)))

    car_features = extract_features(cars,
                                    color_space=color_space, 
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins, 
                                    orient=orient,
                                    pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)
                                    
    notcar_features = extract_features(notcars, 
                                       color_space=color_space, 
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins, 
                                       orient=orient,
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Check the prediction time for a single sample
    t=time.time()
    
    # Save the svc
    pickle.dump([svc, X_scaler], open("svc.p", "wb" ))

    