import matplotlib
matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import cv2
import glob

images = glob.glob("examples/*.png")
print(images)

for image in images:
    print(image)
    img = mpimg.imread(image)

    plt.imshow(img)
    plt.show()