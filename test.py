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


def main():
    # train_svc()
    # test_images()
    test_video_images()
    
def test_make_heat():
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')

    
def test_heat():    
    # Read in the last image above
    image = mpimg.imread('img105.jpg')
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # Display the image
    plt.imshow(draw_img)

    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

    for idx, boxlist in enumerate(bboxes):
        pass
        
    final_map = np.clip(heat - 2, 100, 100)
    plt.imshow(final_map, cmap='hot')   

   
def test_chain():    
    image = mpimg.imread('test_images/bbox-example-image.jpg')
    bboxes = np.array([((100, 100), (250, 200)), ((120, 100), (230, 210)), ((80, 100), (180, 220)),((100, 100), (250, 200)), ((120, 100), (230, 210)), ((80, 100), (180, 220))]) 
    apply_heat(image, bboxes)
    

def test_images():
    images = glob.glob("own_tests/*.png")
    print(images)

    for image in images:
        test_svc_performance(image)
     
     
def test_video_images():
    images = glob.glob("video2images/all/*.png")
    log.info("Number of images: " + str(len(images)))

    svc, X_scaler = pickle.load(open("svc.p", "rb" ))
    
    for image_file in images:
        image = mpimg.imread(image_file)
        window_img, windows = run_svc(image, svc, X_scaler)
        
        root = image_file.split("\\")[-1]
        name = root.split(".")[0] + "_processed.png"
        plt.imsave("video2images/all_out/" + name, window_img)
        
    
def test_train_classifier():
    train_svc()

    test_svc_performance("test_images/bbox-example-image.png")
    test_svc_performance("test_images/img_001.png")    


def test_run_with_heat():
    images = glob.glob("video2images/all/*.png")
    images = sorted(images)
    for image_file in images:
        image = mpimg.imread(image_file)
        
        image_with_windows, heat_window = process_heat(image)
        
        root = image_file.split("\\")[-1]
        name = root.split(".")[0] + "_processed.png"

        plt.imsave("video2images/all_out_heat/" + "heat_" + name, image_with_windows)
       
        gray_image = cv2.cvtColor(image_with_windows, cv2.COLOR_RGB2GRAY)
        float_heat = heat_window.astype(np.float32)
        
        # combined = cv2.addWeighted(gray_image, 1, float_heat, 0.5, 1)
        # plt.imshow(combined, cmap="gray")
        # plt.imshow(image_with_windows)
        # plt.show()


def test_svc_performance(img_file):   
    svc, X_scaler = pickle.load(open("svc.p", "rb" ))

    image = mpimg.imread(img_file)
    window_img, windows = run_svc(image, svc, X_scaler)

    plt.imshow(window_img)
    plt.show()
    
    print(windows)
    
    root = img_file.split("\\")[-1]
    name = root.split(".")[0] + "_processed.png"
    plt.imsave(name, window_img)
        

if __name__ == "__main__":
    # Call the main routine
    main()