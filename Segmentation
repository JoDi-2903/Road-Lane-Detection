import cv2 as cv
from Show_img import *
from matplotlib import pyplot as plt
from Manipulation import warp

def image_segmentation():
    for x in range(8):    
        # Read image, convert to rgb and warp it
        img = cv.imread('./img/Udacity/image00' + str(x + 1) + '.jpg', -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_warped = warp(img)

        # Show images
        plt.figure(figsize=(10,10))
        show_img(plt, 'Original image ' + str(x + 1), img, 2, 1, None)
        show_img(plt, 'Warped image ' + str(x + 1), img_warped, 2, 2, None)
        plt.show()
image_segmentation()