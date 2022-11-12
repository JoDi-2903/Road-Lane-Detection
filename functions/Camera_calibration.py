import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from Show_img import *

# Define columns and rows for chess board
width = 9
height = 6

# Arrays for obj and image points
objpoints = []
imgpoints = []

objp = np.zeros((width*height, 3), np.float32)
objp[:,:2] = np.mgrid[:width, :height].T.reshape(-1, 2)

# Calibrate image
def calibrate_image(x):
    # Read image and convert to gray
    image_path = './img/Udacity/calib/calibration' + str(x + 1) + '.jpg'
    img = cv.imread(image_path, -1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find chess board corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv.findChessboardCorners(img, (width, height), None)
    
    # If corners are found add to obj array -> else return empty image
    if ret == True:
        # Add object points to array
        objpoints.append(objp)

        # Create corners and draw on image
        corners = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        drawn = cv.drawChessboardCorners(img, (width, height), corners, ret)
    else:
        drawn = np.zeros_like(img)
    return drawn

# Undistort image
def undistort_image(img):
    # Convert to gray and calibrate with found points
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)

    # Find new camera matrix and undistort with that matrix
    h, w = img_gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img_gray, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def show_chess_calibration():
    all_images = [None]*20

    # First: calibrate all images
    for x in range(20):
        all_images[x] = []
        all_images[x].append(calibrate_image(x))

    # Then: undistort all images
    for x in range(20):
        image_path = './img/Udacity/calib/calibration' + str(x + 1) + '.jpg'
        img = cv.imread(image_path, -1)
        all_images[x].append(undistort_image(img))

        # Show original, calibrated image and undistorted image
        plt.figure(figsize=(10,10))
        show_img(plt, 'Image ' + str(x + 1), img, 3, 1, 'gray')
        show_img(plt, 'Image ' + str(x + 1) + ' calibrated', all_images[x][0], 3, 2, None)
        show_img(plt, 'Image ' + str(x + 1) + ' undistorted', all_images[x][1], 3, 3, 'gray')
        plt.show()
show_chess_calibration()