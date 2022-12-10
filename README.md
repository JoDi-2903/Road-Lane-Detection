# Road-Lane-Detection
Semester project for digital image processing lecture at DHBW in 2022. 
<br/>
<h2>Task description</h2>

Detect the lane markings or lanes in the [Udacity Nanodegree "Self-Driving Car Engineer"](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013) dataset. Apply the procedures known from the lecture. Feel free to use other procedures from other sources as well. The following objectives must be met on the final submission: 
- **Segmentation**: restrict the image to the area where the track markers are located.
- **Preprocessing**: perform camera calibration (for Udacity image sources) and perspective transformation.
- **Color spaces, histograms**: identify the track markers in the colors of the specified sources. Provided that other lane markers are found on the image, prioritize those lane markers that border your own lane
- **General**: Image processing must take place in real time --> Target: > 20 FPS.
- **General**: Accelerate processing by further measures Consider further measures (e.g. detection of lane marker in first frames, tracking of lane marker in further frames until lane marker positions change too much).
- **Minimal**: relevant track markers are detected continuously in the video "project_video" 
- **Additional**: relevant lane markers are detected in the video "challenge_video" and/or "harder_challenge_video" continuously
- **Additional**: relevant lane markers are applied to the KITTI dataset. What adjustments need to be made for your algorithm to transfer?
- **Additional**: Work out further measures to improve the speed of your algorithm.
- **Additional**: Detect objects in the image and visualize them (e.g. more vehicles, motorcycles, etc.). Please implement the object detection in such a way that it can be deactivated and does not pay into FPS calculation.

<h2>Overview diagrams of the processing steps</h2>

![Screenshot 2022-06-25 183036](https://user-images.githubusercontent.com/88625959/199621038-05e5013c-b362-4cdd-a01d-a5462b39c018.png)

<b>Figure 1:</b> <i>Image processing pipeline</i>
<br/><br/>

![Screenshot 2022-06-25 183036](https://user-images.githubusercontent.com/88625959/199621038-05e5013c-b362-4cdd-a01d-a5462b39c018.png)

<b>Figure 2:</b> <i>Video processing and FPS-Counter</i>
<br/><br/>

<h2>Port to Android App</h2>

A separate GitHub repo has been created for the app, which can be accessed via this [link](https://github.com/Patr1ick/Android-LaneDetection).

<h2>Credits</h2>

x
