# **Advanced Lane Finding Project**

The goal of this project is to identify lane lines on the road using some advanced computer vision algorithms.

There are potential shortcomings with the current pipeline, which are identified, and methods to address them are also elaborated.

### Summary of the pipeline


The pipeline consists of ten steps to find lanes in a single image.

The steps are enumerated below.

* Pre-condition: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Step 1: Apply a distortion correction to raw images.
* Step 2: Use color transforms to create a thresholded binary image.
* Step 3: Use gradients to create a thresholded binary image.
* Step 4: Create a thresholded binary image from both color and gradient filtered binary images.
* Step 5: Apply a perspective transform to warp binary image ("birds-eye view").
* Step 6: Detect lane pixels and fit to find the lane boundary.
* Step 7: Determine the curvature of the lane and vehicle position with respect to center.
* Step 8: Perform sanity check on the intermediate results.
* Step 9: Warp the detected lane boundaries back onto the original image and visual display of the outputs.

[//]: # (Image References)

[image1]: ./writeup_images/straight_lines1.jpg "Original"
[image2]: ./writeup_images/undistorted-straight_lines1.jpg "Distortion Corrected"
[image3]: ./writeup_images/color-combined-straight_lines1.jpg "Color Filtered"
[image4]: ./writeup_images/gradient-combined-straight_lines1.jpg "Gradient Filtered"
[image5]: ./writeup_images/filters-combined-straight_lines1.jpg "Color & Gradient Filters Combined"
[image6]: ./writeup_images/warped-straight_lines1.jpg "Perpespective Transformed (Birds-Eye View) Image"
[image7]: ./writeup_images/windowslider-straight_lines1.jpg "Window-Slider Algorithm Result"
[image8]: ./writeup_images/unwarped-straight_lines1.jpg "Color Filled Lane in Original Image Space"
[image9]: ./writeup_images/output-straight_lines1.jpg "Output Image"
[image10]: ./writeup_images/calibration1.jpg "Distorted Calibration Image"
[image11]: ./writeup_images/undistorted-calibcalibration1.jpg "Undistorted Calibration Image"
[video1]: .output_videos/project_output.mp4 "Output Video"

### The original image

The original image on which the pipeline is applied is shown below.

![alt text][image1]

### Pre-condition: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

Before applying various Computer Vision algorithms, the distortion caused by the lenses of the camera should be removed. For this the camera calibration matrix and distortion coefficients for the used camera should be calculated.

The "object points" will be the (x, y, z) coordinates of the chessboard corners in the world. Here an assumption is made that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are detected.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The  `objpoints` and `imgpoints` are used to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function. The results are applied to the image using the `cv2.undistort()` function. 


In the implementation these processes are wrapped using a class called `Camera`. Once the object is created for the class, the private method `__calibrate_camera()` calculates the calibration matrix and distortion coefficients using the images available in `camera_cal` folder. Once the required parameters are created, an image can be undistorted using the function `undistort_image()`.

One example of the undistorted image is shown below.

#### Distorted Image
![alt text][image10]
#### Undistorted Image
![alt text][image11]

## Pipeline for a single image

### Step 1: Apply distortion correction to raw images.

As a first step in the pipeline, distortion correction is applied to the input image using the interface `undistort_image()` from `Camera` class.

The output from this step is

![alt text][image2]

### Step 2: Use color transforms to create a thresholded binary image.

As the next step, the lanes are identified from various channels in different color channels.

In the RGB color space, the R channel and in HLS color space, the S channel detects the lanes ignoring other details for the given set of test images. The detection of lane pixels are further finetuned using thershold parameters. The output image is a combination pixels identified in R channel and S channel. 

(R and S channels are selected because of the improved lane detection in these channels for the given set of test images. This can differ for another set of images.) 

The above process is accomplished in the class `ColorFilter`. The interface `apply_color_thresholds()` applies the given thresholds on an image and provides a binary output image.

The output from this step is

![alt text][image3]

### Step 3: Use gradients to create a thresholded binary image.

In this step, the gradients are used to detect the lane lines. The gradients over x-axis  and y-axis are calculated and pixels common to both are filtered. Similarly, the magnitude of gradients in x-axis and y-axis are combined and direction of the gradients are also calculated for the given image. The pixels common to both are filtered out. 

The final output image is a combination of pixels identified in both sets as mentioned above. (The selection of pixels (i.e. to `and` or to `or` the filtered images) is done based on the lane detection performance for the given set of test images. This can differ for another set of images.) 

The above process is accomplished in the class `GradientFilter`. The interface `apply_gradient_thresholds()` applies the given thresholds on an image and provides a binary output image.

The output from this pipeline is

![alt text][image4]

### Step 4: Create a thresholded binary image from both color and gradient filtered binary images.

A final binary image is created using the identified pixels from the above two processes.

The output from this pipeline is

![alt text][image5]

### Step 5: Apply a perspective transform to rectify binary image ("birds-eye view").

The next step is to apply perspective transform so that the lanes can seen in the 'birds-eye view'.

The transformation matrix is calculated using the source and destination points. The source points are selected such that only the region containing the lanes is cropped. The destination points are selected such that the lanes in birds-eye view are parallel.

With trail and error, it has been found the following source and destination points satisfies the above conditions. [1]

```python
self.src = np.float32([ [math.ceil(self.width*0.423) , math.ceil(0.652*self.height)], [math.ceil(self.width*0.08), self.height],  \
                          [math.ceil(self.width*0.921), self.height],  [math.ceil(0.578*self.width) , math.ceil(0.652*self.height)] ])
        
self.dst = np.float32([ [math.ceil(self.width*0.156), 0], [math.ceil(self.width*0.156), self.height], \
                               [math.ceil(self.width*0.843), self.height],  [math.ceil(self.width*0.843) , 0] ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 542, 470      | 200, 0        | 
| 100, 720      | 200, 720      |
| 1080, 720     | 1080, 720     |
| 738, 470      | 1080, 0       |


The above process is accomplished in the class `PerpectiveTransformer`. The parameter transformation matrix is calculated for the input image when an object for this class is created. The interface `apply_perpective_transform()` applies perspective transform on a binary image and provides a warped image (which is the birds-eye view in our case).

The output from this pipeline is

![alt text][image6]

### Step 6: Detect lane pixels and fit to find the lane boundary.

As the next step, the lane pixels in the perspective transformed image is identified. For this two algorithms namely window-slider and search-around-polynomial are used. 

The window-slider algorithm identifies lane pixels using small windows. The windows are used to limit the region of interest. The initial position of the window is calculated using histogram peak values. The windows are then moved vertically and horizontally. In vertical direction an arbitrary height of window is selected. In horizontal direction the window is moved according to the mean of the identified pixel indices. 

The identified pixels are then fit with a second degree polynomial (i.e.curve) using the `polyfit` function.

The search-around-polynomial identifies lane pixels by searching within a threshold around a polynomial curve. This is applied only in case of video, where the curve of the lanes does not greatly differ in consecutive frames. The polynomial fit from the previous frame is used and the lanes are searched within a threshold of the given polynomial function. The identified pixels are then fit with a second degree polynomial (i.e.curve) using the `polyfit` function.

The selection of used algorithm for a particular image/frame depends on the following factors.

* If sanity check (see Step 8 below) has failed in the last two frames, apply window-slider algorithm.
* If the given image is the first image/frame, apply window-slider algorithm.
* If a polynomial is available from the last image, use search-around-polynomial algorithm.

The above process is accomplished in the class `LaneFinder`. The interface `get_lane_polynomials()` finds the lane pixels in a given image and returns the lane boundary polynomial. The selected algorithm depends upon the input parametes.

The output from this step (using window-slider algorithm) is

![alt text][image7]

### Step 7: Determine the curvature of the lane and vehicle position with respect to center.


The radius of curvature is found using the formula

**Radius of curvature = (1+(2Ay + B)^2)^(3/2) / |2A|**

where A, B are the coefficients of second and first power respectively in the polynomial function. y is the evaluation plane (where the camera is placed). Here it is the bottom of the image.

The radius of curvature calculated above is in pixel/image space. In order to convert it to real world space, the following constants are used.

```python
        # Define conversions in x and y from pixels space to meters        
        # Based on the perspective transformation the following parameters are defined
        self.ym_per_pix = 23/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/735 # meters per pixel in x dimension     
```

The ym_per_pix is approximated by measuring the length of the lane within the region of interest. The approximation is based upon the fact that each dashed lane is of length 10 feet and the distance between each dashed lane is 30 feet.

The xm_per_pix is approximated by measuring how many pixels exits between the center of left lane and center of right lane in a warped image (i.e. birds-eye view image). The legal width of the lane which is 12 feet is divided by the number of pixels required to represent them in warped image. Both these constants are expressed in meters.

The offset of the vehicle from the center of lane is calculated using the formula

offset = center_lane - center_vehicle 

where 

center_vehicle is the center of the x-axis, 

center_lane is left_lane_pixel + (right_lane_pixel - left_lane_pixel)/2

The left_lane_pixel and right_lane_pixel are determined from the respective polynomials using y value at the bottom of the image (evaluation plane).

### Step 8: Perform sanity check on the intermediate results.

A sanity check is performed on the identified polynomials, radius of curvature and offset for the left and right lanes. 

The sanity check passes only when the following conditions are satisfied. (The last two conditions applies only for video input)

* The width of the lane is within the range 3 - 4.5 meters.
* The lanes are roughly parallel.
* The width of the lane of the current frame differs from the width of the lane of the previous frame within a margin.
* The radius of curvature of the current frame differs from the radius of curvature of the previous frame within a margin.

### Step 9: Warp the detected lane boundaries back onto the original image and visual display of the outputs

In order to reduce the flickering effect on videos, the weighted average of polynomials from last N frames are used for displaying the output images.

The lines for the lanes are plotted using the polynomials. The area between the identified lanes is color filled using the function `cv2.fillPoly()`. 

The color filled image is then warped back to the original image space using the interface `remove_perspective_transform()` of the class `PerspectiveTransformer`. For this process, the function uses inverse transformation matrix calculated from the destination and source points.

Finally, the color filled image is added to the original image (undistorted) and the other details like radius of curvature (average of left and right lane's radius of curvature) and offset are added to the image using the `cv2.putText()` function.

This is done using the function `get_output_image()` in the notebook.

The output from the warping to original image space is

![alt text][image8]

### The output image

The output image from the pipeline is 

![alt text][image9]

---

### Pipeline (video)

In order to keep track of the output from the previous frames, a class called `LaneHistory()` has been used. This class provides an interface called `update_history()` to update the history when the intermediate results has passed the sanity check. The interface `get_average_fit()` can be used to a polynomial which is a weighted average of the history values.

The output video can be found in the link:

Here's a [link to the video result](./output_videos/project_output.mp4)

---

### Discussion

#### Shortcoming identified and resolved

The following shortcomings have been identified and resolved.

* There was a visible flickering effect when the pipeline was used on series of frames aka video. In order to reduce this, the history of previous frames was used to smoothen out outliers.

* There were some radius of curvature values not within a reasonable limit. These outliers are caused by tight fit of the polynomial for the identified lane pixels. Such outliers are removed with the help of sanity check.

#### Shortcoming identified and possible improvements to the pipeline

The following shortcomings have been identified with the current pipeline:

* There exists a huge possibility to finetune the various threshold values in `sanity_check()` function. Currently, arbitrary values are considered as base values and finetuned for the current input image set.

* When there are sharpe curves (as seen in harder challenge video), the window slider algorithm will fail because the vertical movement of windows will not yield result. The reason is that the lane lines will end in the middle of the frame. 
    The window slider algorithm can be improved to handle such scenarios.
   
* Also in sharpe curves, one of the lanes is not completely visible. An algorithm (for e.g. detection based on lane width history) that estimates the lane line should be developed since the current estimation from history will fail because of the greater change in curve polynomails.

* The threshold values selected for various algorithms have been finetuned only for the given set of images and videos. There is a possiblity that these values could either be entirely wrong or be finetuned for another set of scenarios.
    Paricularly the color channels and gradient combinations are currently selected based on the given set of images. These can be further finetuned by using a wide set of input scenarios.
    In `GradientFilter` class, the image can be converted to other color spaces instead of using grayscale image. This can improve lane detection.

* In the challenge video, the region of interest is completely different from the project video. The source points in `PerpectiveTransformer` class is not robust enough to handle the challenge video and will definitely fail in harder challenge video. A new set of source points will require new set of destination points, which are currently selected manually through trail and error.

    An algorithm which automatically calculates the source and destination points based on the identified lane lines from previous frames can be used. This will make the current pipeline generic.


As future work, the shortcomings addressed above can be worked upon to improve the performance of our lane finding pipeline.

### References:

[1] The source and destination points: https://knowledge.udacity.com/questions/307899

[2] The code for the pipeline uses several code snippets from the Udacity coursework and exercises.
