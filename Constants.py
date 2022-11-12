import numpy as np

# For radius calculation (pixels to meters)
y_m_per_pix = 30 / 720
x_m_per_pix = 3.7 / 1280

# number of windows for sliding windows 
nwindows = 180

# Source and destination for perspective transformation
src = np.array([[598, 448], [684, 448], [1026, 668], [278, 668]], np.float32)
dst = np.array([[300, 0], [980, 0], [980, 720], [300, 720]], np.float32)