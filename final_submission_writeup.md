##Advanced Lane Finding
###Jeff Fletcher

---

**Advanced Lane Finding Project**

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

[image1]: ./camera_cal/calibration1.jpg "Original Chessboard"
[image2]: ./camera_cal/test_undist.jpg "Undistorted Chessboard"
[image3]: ./top_down.jpg "Top Down View"
[image4]: ./combined_binary.jpg "Binary Mask"
[image5]: ./test_images/test6.jpg "Original Image"
[image6]: ./histogram_mask.jpg "Histogram Mask"
[image7]: ./curve_fitting.jpg "Curve Fitting"
[image8]: ./result.jpg "Result"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Below is a discussion of each of the rubic points for the project.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code used to compute the camera matrix and distortion coefficients can be found in the file "calibration.py". Here are the steps:

1. Create a matrix of 6x9 object points which will represent the (x,y,z=0) coordinates of the calibration chessboards.
2. Use the openCV function "findChessboardCorners" to identify each chessboard corner, and append the image and real wolrd coordinates of the corner to a multi-dimensional arrays consisting of an array of coordinates for each image.
3. Repeat for each calibration image
4. Once the corners have been identified in each image, calculate the camera matrix and distortion coefficients by feeding the image and real world arrays into the cv2.calibrateCamera function.

The resulting distortion is demonstrated below:

#####Original Chessboard
![alt text][image1]

#####Undistorted Chessboard
![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Here's an example of the distortion correction applied to one of the test images:

![alt text][image3]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I chose to apply a few filters and gradients to create a threshold binary image:

1. HLS Filter (function "hls_select" in "video_pipeline.py", lines 109 - 122): I strongly relied on two HLS filters, one for yellow and one for white, to identify the lane lines. I attempted to give each HLS filter a reasonable range of the color space for the given type of line.

|   |Yellow (min,max)| White (min,max)| 
|:-:|:--------------:|:--------------:| 
| H | (15, 50)       | (0,255)        | 
| L | (140, 190)     | (200, 255)     |
| S | (100, 255)     | (0, 255)       |

2. X and Y Direction Sobel Gradient (function "abs_sobel_thresh" in "video_pipeline.py", lines 40 - 60): I applied a gradient threshold in the x and y direction separately to attempt to identify strong indications of a line that were missed by my color filter. 

A pixel was "true" in the binary image if it (passed the yellow filter) OR (passed the white filter) OR (passed the X Sobel gradient AND the Y Sobel gradient filters).  Here is an example of the original top down image and the resulting binary:

![alt text][image3]
![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the perspective transform discussed above to transform the original image into a top down view in the function "unwrap", lines 16-37 of "video_pipeline.py". In order to do this, I manually selected points on the image corresponding to a straight freeway section for my source points, and arbitratily created destination points to make the top down view a reasonable size. These values can be found in lines 21 - 31 of "video_pipeline.py". Note that different videos and images might require different source points. Here's an example of the image transformation: 

![alt text][image5]
![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used moving histogram windows to identify lane line pixels within the image (lines 183 - 211 of "video_pipeline.py"). Each histogram consisted of half of the image, and the window moved in 1/8 of the image increments until it reached the top of the image. Within each histogram, I identified the maximum value on the left half and right half of the image and built a mask around these values. Once I had completed this for the entire image, I could combine this mask with the thresholded binary image to identify lane line pixels. Below is an example of an overlay of the two masks. The threshold binary image is white and the histogram mask is the red overlay.

![alt text][image6]

Once I had my lane-line pixels, I fit a second order polynomial to the left and right lane points separately (lines 214 - 267 of "video_pipeline.py"). Here's an example of the resulting polynomial:

![alt text][image7]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lane in lines 269 - 277 of "video_pipeline.py". Since the left and right lanes have different curvatures, I decided to estimate a center curvature for the lane. I did this by calculating the midpoint between the lanes at three points, and then fitting a second order polynomial to these midpoints. I then used the midpoint polynomial in the standard radius of curvature calculation.

I calculate the position of the vehicle with respect to the center of the lane in lines 279 - 283 of "vehicle_pipeline.py". The first step was to calculate the center of the vehicle. I assumed that the camera was mounted in the center of the vehicle, and therefore the center of the vehicle was the center of the original image. I then drew a line in the original image and performed the perspective transform to see where the center of the vehicle was in the top down view. I used the pixel value for the warped center location at the bottom of the image (closest to the car) as my starting point. In order to calculate the offset, I compared my estimated left lane line location at the bottom of the image to my warped center location less half the width of the lane line.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I transformed my projection back to the original perspective in lines 285 - 300 of "video_pipeline.py". Here's what the final result looks like:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think my project relies too heavily on my HLS filter. This works very well for the project viedo, but doesn't work well for the challenge videos. I would like to explore sobel filtering in color spaces other than RGB to see if that helps provide more information. I also have limited post post processing in my pipeline. I could incorporate a simple low pass filter to improve smoothness and reject help reject noise in the image. Again, this doesn't have an effect on the project video but would help with the challenges.


