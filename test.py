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

index = 0


def test_images():
    images = glob.glob("test_images/*.jpg")
    print(images)

    for image in images:
        test_svm_performance(image)
     
     
def test_video_images():
    images = glob.glob("video2images/in/*.png")
    log.info("Number of images: " + str(len(images)))

    svc, X_scaler = pickle.load(open("svc.p", "rb" ))
    
    for image_file in images:
        image = mpimg.imread(img_file)
        window_img = run_svm(image, svc, X_scaler)
        
        root = image_file.split("\\")[-1]
        name = root.split(".")[0] + "_processed.png"
        plt.imsave("video2images/out/" + name, window_img)
        
     
def process_video():
    log.info("Running video ...")

    from moviepy.editor import VideoFileClip

    file = "project_video"
    
    in_filename = "./project_videos/" + file + ".mp4"
    log.info("Loading file " + in_filename)
    clip = VideoFileClip(in_filename)
    
    out_filename = "./project_videos/" + file + "_processed.mp4"
    log.info("Writing file " + out_filename)
    
    output_clip = clip.subclip(30, 40).fl_image(process_image)
    output_clip.write_videofile(out_filename, audio=False)
    
    
def process_image(image, frame_name=""): 
    image = image.astype(np.float32)/255.0
    
    # window_img = run_svm(image, svc, X_scaler)
    window_img, dummy = process_heat(image)
    
    window_img = window_img.astype(np.float32)*255.0
    
    return window_img
   
   
def train_classifier():
    train_svm()

    test_svm_performance("test_images/bbox-example-image.png")
    test_svm_performance("test_images/img_001.png")    


def run_with_heat():
    images = glob.glob("video2images/in/*.png")
    
    for image_file in images:
        image = mpimg.imread(image_file)
        
        image_with_windows, heat_window = process_heat(image)
        
        root = image_file.split("\\")[-1]
        name = root.split(".")[0] + "_processed.png"
        plt.imsave("video2images/out/" + name, image_with_windows)
        plt.imsave("video2images/out/" + "heat_" + name, heat_window)
               
        
def process_heat(image):
    occurences = 4
    allowance = 6
    
    window_img, bboxes = run_svm(image, svc, X_scaler)

    all_bboxes.append(bboxes)
   
    if len(all_bboxes) >= allowance:
        all_bboxes.remove(all_bboxes[0])
                
    all_boxes_tuples = []
    for boxes in all_bboxes:
        all_boxes_tuples = all_boxes_tuples + boxes
     
    image_with_windows, heat_window = apply_heat(image, all_boxes_tuples, occurences)
    
    return image_with_windows, heat_window

svc, X_scaler = pickle.load(open("svc.p", "rb" ))

# train_classifier()
# test_video_images()
# process_video()
# run_with_heat()

all_bboxes = []
process_video()
