# This file contains code based on the quizzes for the
# implementation of the heatmap approach.

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label
import logging as log


# Draw the labels as boxes on the image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (1,0,0), 6)
    # Return the image
    return img


# Adds +1 for each detected box in the box list to the heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


# Applies a given treshold to the heatmap
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    # Return thresholded map
    return heatmap


# Applies the heat to the given image an returns the heatmap and clipped image    
def apply_heat(image, bboxes, occurences):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, occurences)

    labels = label(heatmap)

    print("")
    log.info(str(labels[1]) + ' car(s) found')

    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    final_map = np.clip(heatmap, 0, 255)

    debug = False
    if (debug):
        plt.imshow(heatmap, cmap='hot')
        plt.show()
        plt.imshow(labels[0], cmap='gray')
        plt.show()
        plt.imshow(final_map, cmap='hot')
        plt.show()

    return draw_img, final_map

