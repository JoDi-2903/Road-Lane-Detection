import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

####################  Show_img  #########################
# Show image inside defined plot
def show_img(plt, title, img, numCols, pos, cmap):
    plt.subplot(1, numCols, pos)
    plt.title(title)
    plt.imshow(img, cmap)


####################  Manipulation  #########################
# Source and destination for perspective transformation
src = np.array([[598, 448], [684, 448], [1026, 668], [278, 668]], np.float32)
dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)


####################  Segmentation  #########################
# Warp image perspective
def warp(img, src=src, dst=dst):
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))

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