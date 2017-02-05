##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/calibration.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard
is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function.
I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
Provided the camera distortion coefficients from the previous step I used the `cv2.undistort()` method to correct the image distortion.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color ranges and sobel gradient to detect the lane lines. I used a the `cv2.inRange()` method to detect the yellow and white pixels in the image. For the yellow detection,
I converted the image to HSV and used the following a range (20-40, 100-255,100-255). For white color detection I used the RGB image with ranges(10-255,100-255,200-255).
I also use a a sobel filter to get the x gradient. this can be found in `lane_finder.py` from line 63 to 97. Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform is in function called `convert_convert_to_birds_eye_view()` in `lane_finder.py`. The function uses a source
and destination transformation matrix that has the following coeficients.
```
def get_transformation_source(self, img, left=0.44, right=0.56, top=0.655, bottom=1):
    h, w = img.shape[:2]
    top_left = [w * left, h * top, ]
    top_right = [w * right, h * top]
    bottom_left = [w * 0.16, h * bottom]
    bottom_right = [w * 0.86, h * bottom]
    return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)

def get_transformation_destination(self, img):
    h, w = img.shape[:2]
    top_left = [w * 0.25, 0]
    top_right = [w * 0.75, 0]
    bottom_left = [w * 0.25, h]
    bottom_right = [w * 0.75, h]
    return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 563.20, 471.60| 320, 0        |
| 204.80 , 720. | 320, 720      |
| 1100.80, 720  | 960, 720      |
| 716.80, 471.60| 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a polynomial fit to detect the lane lines.  The algorithm starts with a histogram of the image to detect the peaks where the initial line centers are located.
Then it splits the image in windows horizontally and looks for pixels that are part of that window. If the amount of pixels crosses a threshold it updates the image center
so that the next window will start from a different x position. The pixels positions for each lane are then fited with a polynomial and the where they are converted into a
line definition coefficients.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of curvature in lines 164 through 174 in my code in `utils.py`. The curvature is calculated first by converting pixels to meters. then a polynomial
is fitted to these converted pixels which is in meters. For the radius of the curvature we are using the first derivative and second derivative of x with respect to y.

For the deviation from the center we are measuring the center from the image and subtracting that from the center of the image. This can be found in file `lane_finder.py`
in line 17 to 24


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 129 through 153 in my code in `lane_finder.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging thing in the project was to get the binary mask with the least amount pixels that were not part of the road markings. I tried using different gradients but
on images that had shadows or color changes to a lighter color road it was producing incorrect results. Finaly I decided to match the lane colors and use the horizontal gradient
The pipeline will probably fail on images that have many color artifacts due to lighting conditions. It has to be tried during night drving also the horizontal gradient could produce
lines that are not part of the lane so I think the color is more important as means of identifing lanes. Having said that the color can fail if there are other objects that are not
part of the lane which have similar color(white or yellow cars in front)
