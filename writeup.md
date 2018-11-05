## Advanced Lane Finding Project

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
[image1]: ./camera_cal/calibration1.jpg "Calibration Image"
[image2]: ./output_images/undistorted_chessboard.jpg "Undistorted Chessboard Image"
[image3]: ./test_images/test2.jpg "Test Image"
[image4]: ./output_images/undistorted_image.jpg "Undistorted Image"
[image5]: ./output_images/distorted_undistorted_overlapped.jpg "Overlapped Image"

### Setup

##### Link to Jupyter Notebook
![Jupyter notebook][./Advanced_Lane_Lines.ipynb]

### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#### 1.1 Computation of camera calibration matrix and distortion coefficients.

The code comprises of two methods:  
- get_distortion_vars(): This function steps through the list of calibration images, and finds their object points and image points to perform camera calibration.
- undistort(image): This function takes an image, and undistorts it using the calibration attributes returned by the above function.

#### 1.2. Test for image distortion correction on chessboard image
In this section, I have used the functions in code section 1.1 to calibrate and undistort one of the calibration image.  
##### Calibration Image:
![alt text][image1]

##### Undistorted Chessboard Image:
![alt text][image2]

### 2. Apply distortion correction to raw images.
In this section, I have used the functions from Code section 1 to undistort an image of a road image.  

##### Original Image
I have used the image `/test_images/test2.jpg` for the purpose of testing the functionalities below.  
![alt text][image3]

##### Undistorted Image
![alt text][image4]

##### Overlapped Image
The difference between the original and undistorted images isn't quite evident when seen separately. So I have created an overlapped image.  
![alt text][image5]
