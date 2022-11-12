import numpy as np
import cv2 as cv
from Show_img import *
from matplotlib import pyplot as plt
from Manipulation import warp, crop, merge, overlay_alpha
from Filter import canny_space, color_space
from Curve_fitting import calc_curve
from Sliding_windows import sliding_windows
from Apply_area import calc_area, calc_lanes

def calc_kitti_images():
    imgs = ['09', '10', '11', '12', '13', '14', '15']

    # Step 1: Define source and destination for transformation
    src = np.array([[595, 340], [730, 340], [1150, 720], [145, 720]], np.float32)
    dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)

    # Define ROIs for cropping
    ROI = np.array([[(570, 360), (1150, 340), (1150, 720), (1000, 720) , (700, 480), (350, 720), (115, 720)]], dtype=np.int32)
    ROI2 = np.array([[(525, 400), (1150, 400), (1250, 720), (1000, 720) , (700, 480), (350, 720), (115, 720)]], dtype=np.int32)

    # Iterate through all images
    for img_num, img in enumerate(imgs):   
        # Step 2: Read image and convert into rgb
        frame = orig = cv.imread('./img/KITTI/image0' + img + '.jpg', -1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Step 3: Canny edge -> Convert to gray, calculate canny and crop image, then transform result
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        canny = canny_space(gray, 100, 200)
        canny = crop(canny, ROI2)
        canny_t = warp(canny, src, dst) 

        # Step 4: Color space -> Crop image, transform image and apply color filters
        cropped = crop(frame, ROI)
        frame = warp(cropped, src, dst) 
        color = color_space(frame) 
             
        # Step 5: Sliding windows for merged canny and color filters    
        midpoint, lefts, rights = sliding_windows(merge(gray, canny_t, color))
        
        # Step 6: Calculate curve and radius
        left_fit, right_fit, radius = calc_curve(lefts, rights, frame)

        # Step 7: Fill area with color, warp back and overlay with original image. Convert image to RGB
        AOI = calc_area(frame, calc_lanes(left_fit, midpoint, True), calc_lanes(right_fit, midpoint, False), True)
        AOI = warp(AOI, dst,src)
        frame = overlay_alpha(orig, AOI)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Show images
        plt.figure(figsize=(10,10))
        show_img(plt, 'KITTI Image ' + str(img_num + 1) + " (Canny + Color Filter)", merge(gray, canny_t, color), 2, 1, None)
        show_img(plt, 'KITTI Image ' + str(img_num + 1) + " (Kurven Radius: " + str(round(radius)) + ")", frame, 2, 2, None)
        plt.show()
calc_kitti_images()