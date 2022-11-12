import numpy as np
from Constants import nwindows
from Radius import calc_radius

# Calculate curves of lanes with polyfit function
def calc_curve(leftx, rightx, img):
    y = np.linspace(len(img), 0, nwindows+1).astype(int)
    left = np.polyfit(leftx, y, deg=2) 
    right = np.polyfit(rightx , y, deg=2)
    # Calculate radius for curves
    radius = calc_radius(left, right, leftx, rightx)
    return left, right, radius