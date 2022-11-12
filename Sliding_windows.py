import numpy as np
from Constants import nwindows

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
