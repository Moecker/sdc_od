import matplotlib
matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob

from search_classify import *

images = glob.glob("test_images/*.jpg")
print(images)

for image in images:
    test_svm_performance(image)