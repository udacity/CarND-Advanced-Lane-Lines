# Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output created summarizing my experimentation is found in the [writeup](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup.md) for this project.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References:"
[undistort1]: ./output_images/undistort_chess.png "Undistort chess"
[undistort2]: ./output_images/undistort_example.png  "Undistort  image"
[thresh]: ./output_images/binary_threshold.png "Threshold image"
[warp]: ./output_images/warped.png "Threshold image"
[histo]: ./output_images/histogram.png "Histogram image"
[slidewin1]: ./output_images/slide_window.png "Window 1"
[slidewin2]: ./output_images/slide_window2.png "Window 2"
[final]: ./output_images/final.png "Final"
[finalvideo]: ./project_video_output.mp4 "Video Output"

## What was done

You can have a look at my [Jupyter Notebook File](./advanced-lane-finding.ipynb) to find the code of the different steps

The results produced in the different steps will be illustrated below

### 1. Distortion Correction

![alt][undistort1]

![alt][undistort2]

### 2. Gradient Thresholding

![alt][thresh]

### 3. Bird's Eye View *(Perspective Transform)*

![alt][warp]

### 4. Window Search

![][slidewin1]

### 5. Lane Paint

![][final]

## The Output Video

This pipeline was applied on the given video and this is the output

[Final Video Output](./project_video_output.mp4)



## Credit

Blog Posts:

[Advanced Lane By SujayBabruwad](https://medium.com/@sujaybabruwad/advanced-lane-finding-project-6476732dcf18)

[ALF by Nick Hortovanyi](https://medium.com/@NickHortovanyi/advanced-lane-detection-8b98b79b9cac)

StackOverFlow Posts:

https://stackoverflow.com/questions/42907203/color-thresholding-an-hls-jpeg-image

Github:

Lens Distortion correction implementation - https://github.com/letmaik/lensfunpy