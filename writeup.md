## Advanced Lane Finding Project Writeup

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[undistort1]: ./output_images/undistort_chess.png "Undistort chess"
[undistort2]: ./output_images/undistort_example.png  "Undistort  image"
[thresh]: ./output_images/binary_threshold.png "Threshold image"
[warp]: ./output_images/warped.png "Threshold image"
[histo]: ./output_images/histogram.png "Histogram image"
[slidewin1]: ./output_images/slide_window.png "Window 1"
[slidewin2]: ./output_images/slide_window2.png "Window 2"
[final]: ./output_images/final.png "Final"
[finalvideo]: ./project_video_output.mp4 "Video Output"






## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading the write up file.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of [my Jupyter Notebook](./advanced-lane-finding.ipynb) .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted chessboard][undistort1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:![alt text][image2]

When calibrating the  the camera, the camera matrix and distortion coefficient were computed. Which were in turn used with the `cv2.undistort()` to help get the distortion corrected image.

![alt text][undistort2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used 2 kinds of gradient thresholds:

1. Along the X axis.
2. Directional gradient with thresholds of 30 and 90 degrees.

>  This is done since the lane lines are more or less vertical.

I apply the following color thresholds:

1. R & G channel thresholds to emphasize on yellow lanes.
2. L channel threshold so that we don't take into account edges generated due to shadows.
3. S channel threshold since it does a good job of separating out white & yellow lanes.

This can be seen [right here in the notebook](./advanced-lane-finding.ipynb#Gradients-and-color-transforms)

![alt text][thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Thorough examination of the sample images were made, and vertices were drawn and extracted to perform the perspective transform. The polygon with these vertices are drawn on the image below for visualization. The destination points are chosen so that the straight lanes appear parallel in the transformed image. 

Go [right here in the notebook](./advanced-lane-finding.ipynb#Perspective-Transform)

```python
# Vertices extracted manually for performing a perspective transform
bottom_left = [220,720]
bottom_right = [1110, 720]
top_left = [570, 470]
top_right = [722, 470]

source = np.float32([bottom_left,bottom_right,top_right,top_left])

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]

dst = np.float32([bottom_left,bottom_right,top_right,top_left])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 570, 470  |   320, 1    |
| 722, 470  |   920, 1    |
| 1110, 720 |  320, 720   |
| 220, 720  |  920, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A histogram was drawn according to the position of the lane lines on the bird view image as shown in the classroom. The peaks in the histogram indicates the most likely positions of the lane lines according to the image and to each other

![alt text][histo]

With the information on the position using the histogram a **sliding window search** was performed.

I have used 10 windows with a minimum with of 100 pixels.

The x & y coordinates of the non zero pixels were found  and a polynomial is created for these coordinates, which will help the drawing of the lane lines. [Have a look at my code](./advanced-lane-finding.ipynb#Sliding-Window-Search).



![alt text][slidewin1]![alt text][slidewin2]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To get the radius of curvature, I used the formula and material given in the classroom

Since we perform the polynomial fit in pixels and whereas the curvature has to be calculated in real world meters, we have to use a pixel-meter transformation and recompute again. And going through the maths, we get that, The mean of the lane pixels closest to the car gives us the center of the lane. The center of the image gives us the position of the car. The difference between the 2 is the offset from the center.

The radius of curvature and the offset are computed [Here](./advanced-lane-finding.ipynb#Radius-of-curvature-and-center-offset) in my code.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[Inverse transform](./advanced-lane-finding.ipynb#Inverse-transform)

![alt text][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output_.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

##### Issues and challenges

###### Challenges

**Gradient Thresholding:**

I had to do a lot of experiments with gradient and color channel thresholding trying to tweak the parameters.

The lane lines in were either too bright or too dull, this had me tweaking the channels to obtain Red and Green thresholding with L and S thresholding.

######  Issues

- Bad frame rendering.
- Color thresholding.

##### Failures

I haven't tested the pipeline on the two challenges, but I had a look on both videos, I assume that the pipeline will fail on both of these videos for these reasons

1. The challenge video has much more bad frame rendering than this one
   - *Possible Solution:* we may have to give a better average of the lanes, this will cover up for the missing lanes.
   - *Possible Solution2:* we can look for a way to search and predict bad frames, so as to use good frames already collected as a replacement. However, this solution may have a short coming in the harder challenge because of it's strong turns
2. The challenge videos have a very diverse lane line nature, this will result in even better tweaking of the color channels
3. The polygon taken as region of interest is much big, this won't probably work for the harder challenge video with it's sharp turns and obstacles.