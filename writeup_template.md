## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_steps/straight_lines1_undist.jpg "Undistorted"
[image2]: ./output_steps/straight_lines1_undist.jpg "Road Transformed"
[image3]: ./output_steps/binary_combo.jpg "Binary Example"
[image4]: ./output_steps/perspective.jpg "Warp Example"
[image5]: ./output_steps/fit_poly.jpg "Fit Visual"
[image6]: ./output_steps/final.jpg "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps under the second element of my steps list in example.ipynb).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears under the third element of my steps list in the notebook `example.ipynb` (examples/example.ipynb). The `warper()` function takes as input an image (`img`). The function uses local variables, source (`src`) and destination (`dst`) points to do the transform. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32((180, img.shape[0]), (550, 476), (740, 476), (1125, img.shape[0]))

dst = np.float32((300, img.shape[0]-1), (300, 1), (img.shape[1]-300, 1), (img.shape[1]-300, img.shape[0]-1))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 300, 719      | 
| 550, 476      | 300, 1        |
| 740, 476      | 980, 1        |
| 1125, 720     | 980, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used two approaches to detect the pixels of the binary image which are supposed to be part of the lane lines. Windowing and gates. Windowing was used to initialize the 2nd order polynomial of each line (left and right). Once initialized, gates are built aroung the two estimated polnomials and the search of the pixels is done in the vicinity. 
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the cell under the 5th element of my steps list in my notebook example.ipynb

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the functions warp_back() and write_text(), (6th and 7th elements of my steps list respectively)
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
I used a very basic technique to smooth out my results. I stored the poly parameters of each side (recent 5 estimations), and then took the average over those and built my gates around the averaged poly. It worked well in most of the frames of the project video, but is showed some minor failures when shadows were presented in the frames.
As improvement, tracking the parameters of the polynomials could improve the results.

