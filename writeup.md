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
* Use created pipeline to annotate example video

[//]: # (Image References)

[image0]: ./camera_cal/calibration3.jpg "Distorted"
[image1]: ./output_images/undistorted_calibration3.jpg "Undistorted"
[image_undist]: ./test_images/straight_lines1.jpg "Distorted Road"
[image2]: ./output_images/undistorted/straight_lines1.jpg "Undistorted Road"
[image3]: ./output_images/binary/test2.jpg "Binary Example"
[image4]: ./output_images/bird/test2.jpg "Warp Example"
[image5]: ./output_images/poly/test2.jpg "Fit Visual"
[image6]: ./output_images/lane/test2.jpg "Output"
[video1]: ./output_video/project_video_with_overlay.mp4 "Video"

### All code for this project is located inside IPython notebook "Advanced-Lane-Lines.ipynb"


### Camera Calibration

#### The code for this step is contained in the code cell with comment "cell 1" under "Camera calibration" block

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test images.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original image          |  Undistorted image      
:-------------------------:|:-------------------------:
![alt text][image0]  |  ![alt text][image1]

### Pipeline (single images)

#### 1.Distortion correction for road test image
#### The code for this step is contained in the code cell wih comment "Example of undistorted image" under "Perspective transform" block

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images(effect is clear on the sides of car hood):

Original image          |  Undistorted image      
:-------------------------:|:-------------------------:
![alt text][image_undist]  |  ![alt text][image2]

#### 2.Obtaining thresholded binary image 
#### The code for this step is contained in the code cell wih comment "cell 3" under "Thresholded binary image" block

I used a combination of hls color and gradient thresholds(with help of sobelx transformation) to generate a binary image. Code fot this you can find in `thresholded_binary` method. Here's an example of my output for this step:

![alt text][image3]

#### 3. Obtaining image with perspective transform(birds-eye view)
#### The code for this step is contained in the code cell wih comment "cell 4" under "Perspective transform" block

The code for my perspective transform includes a function called `bird_perspective`.  The `bird_perspective` function takes as inputs an image (`image`), and use (`src`) and destination (`dst`) points(which included into body of a method) to transform image with help of `cv2.warpPerspective` function.  These points we chosen manually for best fit:

```python
    src = np.float32([[587, 450], [700, 450], [1200, img_size[1]], [160, img_size[1]]])
    dst = np.float32([(450,0),
                      (w-450,0),
                    (w-450,h),
                      (450,h)])
```

where `h` and `w` are

```
    img_size = (image.shape[1], image.shape[0])
    
    h = img_size[1]
    w = img_size[0]

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 587, 450      | 450, 0        | 
| 700, 450      | image width - 450, 0      |
| 1200, image height     | image width - 450, image height      |
| 160, image height       | 450, image height        |

Result of these manipulation you can see below(this is test_images/test2.jpg):

![alt text][image4]

#### 4. Finding lane-line pixels and fit their positions with a polynomial
#### The code for this step is contained in the code cell wih comment "cell 5" under "Identify lane-line pixels" block

On this step I created method `find_lane_pixels` where with help of sliding window I found all point belong to left and right lanes(on this step almost everythins else was remove from the image). Method returns arrays with x's and y's of finded lanes. After that in other method(`fit_polynomial`) we are finding polynominal functions which can represent our curved lines on the road. Result of these manipilatons is here:

![alt text][image5]

#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.
#### The code for this step is contained in the code cell wih comment "cell 6" under "Measuring curvature" block

On this step I was using polynaminal lane functions from previous step to calculate curvature. Method which I used were `measure_curvature_real`(this method return cunverted from pixels to meters curvature) and `center_distance` which culculate offset of our car from line center in meters.

#### 6. Example of identified lane area.
#### The code for this step is contained in the code cell wih comment "cell 7" under "Drawing" block

On this step I was using all data received from previous steps to provide overlay on test image with informaton about lane curvature, center offset and lane itself.
`draw_lane_lines` function contain actual pipeline for this overlay. I want to mention that on this step we need transform our perspactive back to get proper result.
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Apply pipeline on video

Overlay pipeline was applied to the test video.
Here's a [link to my video result][video1]

---

### Discussion

During implementaton of this project I was using different new technics to find lane lines. I was using diffrent color spaces, adjust my camera for distortions, warp impage to bird view etc. All these combined turn into good result for identifying road line. But I still think that pipeline could have some improvements. For example, thresholded inmage can use more underlying layers(from other color spaces for example, or better/faster edge detection algorithm). I think my implementation will likely fall at night or poor light condition. Also very big curvature can confuse my algorithm. 
