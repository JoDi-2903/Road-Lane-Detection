import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

####################  Apply_filter  #########################
# Calculate points for lanes
def calc_lanes(street_area, midpoint, isleft):
    # Calculate extreme value for curve
    # DGL: ax^2 + bx + c -> 2ax + b = 0 -> x = (-b/2)/a
    x_extreme = (-street_area[1]) / 2 / street_area[0]

    # Set lower and upper boundaries
    if(isleft):
        lower = 50
        upper = midpoint
    else: 
        lower = midpoint
        upper = 1200
    
    diff_lower = abs(lower - x_extreme) # distance left to extreme point
    diff_upper = abs(upper - x_extreme) # distance right to extreme point
    
    
    # Define x values
    # if left is closer to extreme point -> set extreme point as lower boundary
    if (diff_lower < diff_upper):
        x = np.linspace(x_extreme, upper, 4500).reshape(-1, 1)
    # if right is closer to extreme point -> set extreme point as upper boundary
    else:
        x = np.linspace(lower, x_extreme, 4500).reshape(-1, 1)

    # Get y values from polyfit function
    y = (street_area[0] * (x**2) + street_area[1] * x + street_area[2]).reshape(-1, 1)

    # Merge points -> [[p1], [p2]...]
    xycoor = np.concatenate((x,y), axis=1)
    return xycoor

# Create area between lanes
def calc_area(img, left_points, right_points, with_alpha=False):
    # If with_alpha is True -> Make image colored
    if(with_alpha):
        zero_img = np.zeros((len(img), len(img[0]), 3))
    else:
        zero_img = np.zeros((len(img), len(img[0])))

    # sort by y values
    left_points = left_points.astype(np.int32)[np.argsort(left_points[:, 1])]
    right_points = right_points.astype(np.int32)[np.argsort(right_points[:, 1])][::-1]

    # Merge points -> [[p1], [p2]...]
    points = np.concatenate((left_points, right_points))

    # If with_alpha is True -> color image green, else color it white
    if(with_alpha):
        cv.fillPoly(zero_img, [points], color=[0,255,0])
    else:
        cv.fillPoly(zero_img, [points], color=255)
    return zero_img


####################  Curve_fitting  #########################
# number of windows for sliding windows 
nwindows = 180

# Calculate radius for two lines
def calc_radius(left_fit, right_fit, leftx, rightx):
    global last_r
    # Functions for curves by polyfit
    left_fitx = left_fit[0]*leftx**2 + left_fit[1]*leftx + left_fit[2]
    right_fitx = right_fit[0]*rightx**2 + right_fit[1]*rightx + right_fit[2]
    
    # Calculate functions again with meters for values
    left_fit_curve = np.polyfit(leftx*x_m_per_pix, left_fitx*y_m_per_pix, deg=2)
    right_fit_curve = np.polyfit(rightx*x_m_per_pix, right_fitx*y_m_per_pix, deg=2)
    
    # radius of curve
    left = cal_r(left_fit_curve, leftx)
    right = cal_r(right_fit_curve, rightx)

    # Take average from last radius and current radius (if last_r is not None)
    if (last_r != None):
        radius = round(np.mean([left, right, last_r]),0)
    else:
        radius = round(np.mean([left, right]),0)

    # Save current radius for next calculation
    last_r = round(np.mean([left, right]),0)
    return radius

# Calculate curves of lanes with polyfit function
def calc_curve(leftx, rightx, img):
    y = np.linspace(len(img), 0, nwindows+1).astype(int)
    left = np.polyfit(leftx, y, deg=2) 
    right = np.polyfit(rightx , y, deg=2)
    # Calculate radius for curves
    radius = calc_radius(left, right, leftx, rightx)
    return left, right, radius


####################  Sliding_windows  #########################
def sliding_windows(warped_img, margin=200, minimum=30):
    # Histogram for image
    hist = np.sum(warped_img[warped_img.shape[0]//2:, :], axis=0)
        
    # Take peaks from left and right side of histogramm for starting points and add half margin
    mid = np.int(hist.shape[0] // 2)
    leftx_start = np.argmax(hist[:mid]) - margin // 2
    rightx_start = np.argmax(hist[mid:]) + mid + margin // 2

    # Window height based on number of windows
    window_height = np.int(warped_img.shape[0] // nwindows)
    
    # Calc points that are not zero in images
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Initialize current positions for windows
    leftx_current = leftx_start
    rightx_current = rightx_start

    # Initialize values to be returned -> centers of windows
    lefts_good = np.empty(shape=(1,1), dtype=int)
    rights_good = np.empty(shape=(1,1), dtype=int)

    # Go through every window
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window + 1) * window_height
        win_y_high = warped_img.shape[0] - window*window_height
        
        # Calculate boundaries of the window
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low =  rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
        # Identify the pixels that are not zero within window
        left_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        right_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # If more than minimum pixels are found -> recenter next window
        if len(left_inds) > minimum:
            leftx_current = np.int(np.mean(nonzerox[left_inds]))
        if len(right_inds) > minimum:
            rightx_current = np.int(np.mean(nonzerox[right_inds]))

        # Add centers of windows to array
        lefts_good = np.append(lefts_good, leftx_current)
        rights_good = np.append(rights_good, rightx_current)
    return mid, lefts_good, rights_good


####################  Filters  #########################
# Kernels for filter operations
kernel_small = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')
kernel_large = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], 'uint8')

# Convert image to yellow and white color space
def color_space(img):
    # Convert image to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Colorspace "yellow" in HSV: (15-40, 80-255, 160-255)
    mask_yellow = cv.inRange(img_hsv, (15, 80, 160), (40, 255, 255))
    # Colorspace "white" in HSV: (0-255, 0-20, 200-255)
    mask_white = cv.inRange(img_hsv, (0, 0, 200), (255, 20, 255))
    # Merge white and yellow masks
    masks = cv.bitwise_or(mask_yellow, mask_white)
    # Return image in gray
    return cv.cvtColor(cv.bitwise_and(img, img, mask=masks), cv.COLOR_BGR2GRAY)

# Apply canny filter to image
def canny_space(img, lower=80, upper=200):
    # Equalize histogramm with gray image
    equalized_gray_image = cv.equalizeHist(img)
    # Blur image three times
    equalized_gray_image = cv.GaussianBlur(equalized_gray_image, (5, 5), 0)
    equalized_gray_image = cv.GaussianBlur(equalized_gray_image, (5, 5), 0)
    equalized_gray_image = cv.GaussianBlur(equalized_gray_image, (5, 5), 0)
    # Return image with canny filter
    return cv.Canny(equalized_gray_image, lower, upper) 

# Dilate image (Expands the shapes of the image)
def dilate(img, iterations):
    return cv.dilate(img, kernel_small, iterations=iterations)

# Erode image (Reduces the shapes of the image)
def erode(img, iterations):
    return cv.erode(img, kernel_small, iterations=iterations)

# Close image (Removes noises)
def close(img, iterations):
    return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_large, iterations=iterations)


####################  Manipulation  #########################
# Source and destination for perspective transformation
src = np.array([[598, 448], [684, 448], [1026, 668], [278, 668]], np.float32)
dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)

# Warp image perspective
def warp(img, src=src, dst=dst):
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(img, M, (img.shape[1], img.shape[0]))

# Crop image for region of interest
def crop(img, ROI):
    # Create blank img with same size as input img
    blank = np.zeros(img.shape[:2], np.uint8)

    # Fill region of interest
    region_of_interest = cv.fillPoly(blank, ROI, 255)

    # Create image of interest with region (resize)
    return cv.bitwise_and(img, img, mask=region_of_interest)

# Merge to masks
def merge(frame, img1, img2):
    both = frame.copy()
    both[np.where(np.logical_and(img1==0, img2==0))] = 0
    return both

# Overlay two images
def overlay(img, overlay):
    img[np.where(overlay!=0)] = [0,255,0]
    return img

# Overlay two images with alpha
def overlay_alpha(img, overlay):
    return cv.addWeighted(img, 1, overlay.astype(np.uint8), 0.5, 0.0)

# Show image inside defined plot
def show_img(plt, title, img, numCols, pos, cmap):
    plt.subplot(1, numCols, pos)
    plt.title(title)
    plt.imshow(img, cmap)

####################  KITTI  #########################
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