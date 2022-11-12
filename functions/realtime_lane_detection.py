import cv2 as cv
import numpy as np
import time
from Car_detection import detect_cars
from Manipulation import warp, crop, overlay
from Filter import color_space, close, dilate
from Sliding_windows import sliding_windows
from Curve_fitting import calc_curve
from Apply_area import calc_area, calc_lanes
from Car_detection import highlight_cars

# Open video file
video_file = 'project'
cap = cv.VideoCapture('./img/Udacity/' + video_file + '_video.mp4')

# Manually turn car detection on or off
CAR_DETECTION = False

# If challenge video is played -> Define different points for transformation 
CHALLENGE = (video_file == 'challenge') 
if(CHALLENGE):
  src = np.array([[600, 480], [760, 480], [1026, 700], [278, 700]], np.float32)
  dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)

# Check if camera opened successfully
if (cap.isOpened() == False):
  print("Error opening video stream or file")

# Start timer for fps counter
start_timer = time.time() - 0.01
frame_count = -1

# Read every frame
while(cap.isOpened()):
  ret, frame = cap.read()

  # Check if there is another frame
  if (frame is None):
    break
  orig = frame.copy()

  # Calculate Frame rate
  frame_count += 1
  ellapsed_time = time.time() - start_timer
  frame_rate = frame_count / ellapsed_time  

  # Calculate curve area only every fifth frame for performance reasons
  if (frame_count % 5 == 0):
    # Detect cars if car detection is on
    if (CAR_DETECTION):
      cars = detect_cars(frame)

    # Step 1: Warp image
    frame = warp(frame, src, dst)
   
    # Step 2: Apply color mask and close image for get rid of small distrubances
    color = color_space(frame)
    color = close(color, 10)

    # For challenge video: dilate and crop color mask
    if(CHALLENGE):
      color = dilate(color, 10)
      ROI = np.array([[(525, 360), (1150, 340), (1150, 720), (900, 720) , (700, 550), (350, 720), (115, 720)]], dtype=np.int32) 
      color = crop(color, ROI) 
    
    # Step 3: Sliding windows to get curve points    
    midpoint, lefts, rights = sliding_windows(color)

    # Step 4: Calculate curve and radius
    left_fit, right_fit, radius = calc_curve(lefts, rights, orig)
  
    # Step 5: Fill area and transform area back
    AOI = calc_area(frame, calc_lanes(left_fit, midpoint, True), calc_lanes(right_fit, midpoint, False), False)
    AOI = warp(AOI, dst, src)
  
  # Overlay street area on every image
  frame = overlay(orig, AOI)

  # If car detection is on: Draw rectangles around cars
  if (CAR_DETECTION):
    frame = highlight_cars(frame, cars)

  if ret == True:      
    # Add frame rate to video
    cv.putText(frame, "FPS: "+ str(round(frame_rate)), (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    
    # Show curve radius (not for challenge video)
    if(not CHALLENGE):
      cv.putText(frame, "Curve radius: " + str(round(radius)).replace(".0", "") + "m", (len(frame[0])//2-150, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv.LINE_AA)
    cv.imshow('Frame', frame)

    # Close video with letter 'q'
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
  else: 
    break

# When everything done, release the video capture object
cap.release()
cv.destroyAllWindows()