# Advanced Lane Ling Finding


---

**Advanced Lane Finding Project**

## Project Description
The project consists in detecting the right and left lanes in a video of a car in a highway

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Please run the notebook AdvLaneLines.ipynb to see all the images outputs.
The video is located in test_videos_output


### Camera Calibration

Functions:
- camera_cal()
- save_dist_coeff(img,objpoints,imgpoints)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

In the function 'save_dist_coeff(img,objpoints,imgpoints)', I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.


### Pipeline (single images)

#### 1. Distortion-corrected image.

See notebook - function cal_undistort

#### 2. Thresholded binary image
See notebook - function transform_thresholded(image)

After multiple trials, I found that combining Red filtering and S filtering is efficient to detect yellow lines and partly remove shadows


#### Perspective transform
See notebook - function transform_thresholded(image)

I chose to hardcode the source and destination points in the following manner:

y_bottom = 720
y_top = 450
src_x1 = 240 #bottom left
src_x2 = 1040 #bottom right
src_x3 = 600 #top left
src_x4 = 680 #top right

shrink_base_factor = 0.55

dest_x1 = src_x1 + (src_x2 - src_x1) * (1- shrink_base_factor) / 2
dest_x2 = src_x2 - (src_x2 - src_x1) * (1- shrink_base_factor) / 2


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### Detect lane pixels and fit to find the lane boundary.

2 methods are available:
- function search_from_scratch(): it uses a histogram for detecting the x starting points and then a sliding window method as detailed in the course
- function search_around_poly(): it uses the former polynomial fit to define a search range (+/- 75 pixels)

Then, after detecting the lane pixels, I fit a polynomial of order 2 to define each lane.


#### Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature of the left and right lanes is calculated in the function measure_curvature_real(ploty, left_fit_cr,right_fit_cr).

The position of the vehicle is calculated as the difference between the center of the image (x = 640) and the center of the position of each lane (saved in left_lane.base_pos and right_lane.base_pos)


### Pipeline (video)

The video is located in test_videos_output/

The pipeline (video) is repeating the steps described in the pipeline (single image) with the following changes:
* Lane class created to save characteristics of the left and right lane detections
* sanity checks for lane detection, i.e. the minimum number of pixels detected, differences of curvature (I found the values too volatile to be reliable), distance between 2 lines within reasonable range, parallelism (based on 2 points). Above a certain number of consecutive undetected lanes, the search method becomes search_from_scratch.
* smoothing: I use the last 5 iterations to draw the polynom (I average the fitting values)
* display: I use the function display_text_over_image(img,text,position) to display specific indicators on the image


### Discussion

* The pipeline will fail if no lanes are detected for the 1st frame (could be solved by initializing the values of left_lane and right_lane)

* The pipeline would fail in non-optimal conditions, like bad weather (snow, rain) or bad lighting.

* The pipeline would also probably fail in case of bumpy roads or hard turns

* The shades are a real issue - please provide any guidance to optimize lane detections in case of shades.

* We could define a confidence level indicator and weigh the fitting values accordingly (trade-off smoothing and speed / reactivity in case of changes of curvatures or hard turns for example)

