# **Advanced Lane Finding Project** 

## ChangYuan Liu

### This document is a brief summary and reflection on the project.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image0]: ./output_images/undist_cal.png
[image1]: ./output_images/1_undist.png
[image2]: ./output_images/2_threshold.png
[image3]: ./output_images/3_perspective_transform.png
[image4]: ./output_images/4_find_lanes.png
[image5]: ./output_images/5_plot_lanes.png

---


### 1. Camera Calibration

First, define the camera_calibration function to calculate mtx and dist matrices from the calibration images.

Then, these matrices are saved into a pickle file and used to undistort images later in the pipeline later.

Here is an example to undistort one of the calibration imgages:

![alt text][image0]



### 2. The Pipeline to Process Images

My pipeline consists of 5 steps: 

(1) Apply a distortion correction to raw images.

![alt text][image1]

(2) Use the combination of color transforms and gradients to create a thresholded binary image.

![alt text][image2]

(3) Apply a perspective transform to rectify binary image ("birds-eye view").

![alt text][image3]

(4) Detect lane pixels and fit to find the lane boundary.

![alt text][image4]

In this step, the lane lines found by the program are sensitive even I apply sliding window, so I add an filter to the polyfit coeffients. It's not the best resolution and doesn't work well with sharp turns, but it improves the stability of the lane line postions.

(5) Determine the curvature of the lane and vehicle position with respect to center. Warp the detected lane boundaries back onto the original image.

![alt text][image5]



### 3. Identify potential shortcomings with my current pipeline

One potential shortcoming would be the program wouldn't correctly identify the lane lines when the driving environment changes over time, such as road surface, light conditions, etc. under the situation in the challenge videos.

Another shortcoming is that hyperparameters in the pipeline are manually tuned, which requires a lot of efforts.


### 4. Possible improvements

A possible improvement would be not using global variables to save the previous polyfit parameters as I did in my project.

Another potential improvement would be designing better approaches to handle abnormal situations when there is not lane(s) found.
