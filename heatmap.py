import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.measurements import label


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
    
    
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    # print(len(heatmap[heatmap <= threshold]))
    # print(len(heatmap[heatmap <= 1]))
    
    heatmap[heatmap <= threshold] = 0
    
    # Return thresholded map
    return heatmap


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

    
def apply_heat(image, bboxes, occurences):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    
    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, occurences)

    labels = label(heatmap)

    print(labels[1], 'car(s) found')
    # plt.imshow(labels[0], cmap='gray')
    # plt.show()
    
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    final_map = np.clip(heatmap, 0, 255)
    # plt.imshow(final_map, cmap='hot')
    # plt.show()
    return draw_img, final_map

        
def make_heat():
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')

    
def test_chain():    
    image = mpimg.imread('test_images/bbox-example-image.jpg')
    bboxes = np.array([((100, 100), (250, 200)), ((120, 100), (230, 210)), ((80, 100), (180, 220)),((100, 100), (250, 200)), ((120, 100), (230, 210)), ((80, 100), (180, 220))]) 
    apply_heat(image, bboxes)
    