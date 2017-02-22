# This file is the main starting point of the pipeline.
# It reads in the video, triggers a search of sliding windows
# for each frame and applies the heatmap approach.

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob

import logging as log
import sys
log.basicConfig(stream=sys.stderr, level=log.DEBUG)
import pickle

from search_classify import *
from heatmap import *

# Load the previously trained svc
svc, X_scaler = pickle.load(open("svc.p", "rb" ))

# Keeps track of all detected bboxes
all_bboxes = []

# Parameters for the heatmap outlier detection
kRequiredOccurences = 30
kKeepBoxesIterations = 20

index = 0

# The main entry point when calling the script
def main():
    # train_svc()
    process_video()


# The chain for processing a video.
# Loads the video a calls the process_image on every frame
def process_video():
    log.info("Running video ...")

    from moviepy.editor import VideoFileClip

    # file = "project_video"
    file = "test_video"

    in_filename = "./project_videos/" + file + ".mp4"
    log.info("Loading file " + in_filename)
    clip = VideoFileClip(in_filename)

    out_filename = "./project_videos/" + file + "_processed.mp4"
    log.info("Writing file " + out_filename)

    # output_clip = clip.subclip(15, 17).fl_image(process_image)
    # output_clip = clip.subclip(37, 40).fl_image(process_image)
    # output_clip = clip.subclip(43, 45).fl_image(process_image)
    output_clip = clip.fl_image(process_image)

    output_clip.write_videofile(out_filename, audio=False)


# Processes each frame of a video (or standalone image)
def process_image(image, frame_name=""):
    # Converts to range 0->1 for our chain.
    # Needed, since the video clip produces the range 0-255
    image = image.astype(np.float32) / 255.0

    # Call the sophisticated heatmap chain which runs the svc and keeps track
    # of previously detected bboxes
    window_img, unused_heatmap_image = process_heat(image)

    # Some debug implementation
    debug = False
    if (debug):
        window_img = image

        global index
        if ((index % 20) == 0):
            plt.imsave("img_" + str(index), window_img)
        index += 1

    # Rescale the image
    window_img = window_img.astype(np.float32) * 255.0

    return window_img


# Processes an image with the heatmap info of previously detected bboxes
def process_heat(image):
    # First get the image and the bboxes
    unused_image_with_boxes, bboxes = run_svc(image, svc, X_scaler)

    # Add retreived boxes to boxes list
    all_bboxes.append(bboxes)

    # Very basic circular buffer
    if len(all_bboxes) >= kKeepBoxesIterations:
        all_bboxes.remove(all_bboxes[0])

    # Required conserion from list to tuple
    all_boxes_tuples = []
    for boxes in all_bboxes:
        all_boxes_tuples = all_boxes_tuples + boxes

    # Apply the heatmap chain on extracted list of boxes
    image_with_windows, heat_window = apply_heat(image, all_boxes_tuples, kRequiredOccurences)

    return image_with_windows, heat_window


if __name__ == "__main__":
    # Call the main routine
    main()

