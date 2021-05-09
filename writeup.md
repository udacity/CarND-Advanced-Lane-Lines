# Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholds binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[original]: ./test_images/test5.jpg "Original"
[distortion]: ./output_images/02_distortion/test5.png "Undistorted"
[binary]: ./output_images/03_color_binary/test5.png "Binary"
[trapezoid]: ./doc/trapezoid_straightroad.png "Trapezoid"
[trapezoid_be]: ./doc/trapezoid_birdsEye.png "Trapezoid_birdsEye"
[transform]: ./output_images/04_transform/test5.png "transform"
[sliding]: ./output_images/05_sliding_window/test5.png "sliding_window"
[warped]: ./doc/transform.png "transform"
[findLines]: ./doc/find_lines.png "find_lines"
[curvature]: ./doc/curvature_eq.png "curvature_eq"
[lane_width]: ./doc/lane_width.png "lane_width"
[dashed_line]: ./doc/dashed_line.png "dashed_line"
[output]: ./output_images/06_result/test5.png "output"

## Camera Calibration

The code for this step is contained in **"./code/calibration.py"**
```python
    def calc_calibration_parameter(visuOn = True, writeOn = True)
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The chessboard corners are detected by applying `cv2.findChessboardCorners()`  on a grayscale image with the chessboard size defined by `nx` and `nx`. For each successful detected chessboard the output is written to: **"output_images\01_calibration"**

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. calculate the binary image

## Pipeline (single images)

The code for this step is contained in **"./code/pipeline.py"**
```python
    def run(img, fname = None, visuOn = False, writeOn = False):
```
This run function gets an image as input and then applies following steps to detect the ego lane, compute curvature and calculate the ego lane:
1. undistort the image
2. Detect edges in color and grayscale image
3. Perspective transform 2D image to birdsEye
4. If no line was detected before:
    1. Find lines newly with sliding window
    2. else find lines within previous areal (tracking)
5. Calculate lane curvature and ego position
6. Draw results onto image

I will here explain each step and show the output for one test image.
![][original] 


### 1. Distortion correction

The code for this step is contained in **"./code/distortion.py"**
```python
    def undistort_image(fname, img, visuOn = True, writeOn = True):
```
The once computed camera matrix and distortion coefficients are read from **"code\wide_dist_pickle.p"**. With this input the distortion correction is executed by using the `cv2.undistort()` function. The undistorted images are saved into: **"output_images\02_distortion"**

Original             |  Undistorted
:-------------------------:|:-------------------------:
![][original]  |  ![][distortion]

### 2. Detect edges in color and grayscale image

The code for this step is contained in **"./code/binary.py"**
```python
    def create_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
```
First I convert the input image into HLS color space. Onto the lightness channel the sobel edge operator is applied in x direction only. The output is scaled and threshold is applied. Additionally onto the saturation channel another threshold is applied. Both color and gradient thresholds are combined. The binary images are saved into: **"output_images\03_color_binary"**

Undistorted             |  Binary
:-------------------------:|:-------------------------:
![][distortion]  |  ![][binary]

### 3. Perspective transforms 2D image to birdsEye 

The code for this step is contained in **"./code/transform.py"**
```python
    def transform_and_warp(undist, src, dst):
```
The `transform_and_warp()` first calculates out of the source and destination points the transform matrix M. With M then the `warpPerspective()` warps perspectively the image into birdsEye view image. Therefore it was required to get the correct source and destination points. I defined the points as:

```python
# create symmetric trapezoid as source
src = np.float32([590,450],         #topLeft
                [690,450],          #topRight
                [180,720],          #bottomLeft
                [1100,720]])        #bottomRight

# create vertical lines in image (birdsEye)
dst = np.float32([[250, 100],       #topLeft
                [1030, 100],        #topRight
                [250, 720],         #bottomLeft
                [1030, 720]])       #bottomRight
```
The source points are gathered from a straight road undistorted image by assuming a symmetric trapezoid around the ego lane as ROI.
![][trapezoid]

The destination points are the wished positions of the line markings of straight ego lane in the birdsEye image:
![][trapezoid_be]

Within the pipeline the previously generated binary image is transformed. The warped images are saved into: **"output_images\04_transform"**

Binary             |  Warped
:-------------------------:|:-------------------------:
![][binary]  |  ![][transform]


### 4. Finding lines
The code for this step is contained in **"./code/findingLines.py"**

#### 4a. Find lines newly with sliding window
The code for this step is contained in **"./code/transform.py"**
```python
    def sliding_window(fname, warped, left_line, right_line, visuOn = True, writeOn = True):
```
If no line has been detected before then we need to search newly in the warped binary image fore lines. Therefore the binary image is transformed into grayscale image. Within the bottom half of the image the left maximum of the histogram is chosen as the starting point. (vise versa for the right line). Then stepwise within a defined margin the next left and right lane center is calculated as the mean of the new window. By that all the relevant pixels for the left and right line are found.
Those points are then fitted into a polygon 2nd degree. The visualization of the found lines are saved into: **"output_images\05_sliding_window"**

Warped             |  Sliding_window
:-------------------------:|:-------------------------:
![][transform]  |  ![][sliding]

#### 4b. Find lines within previous areal (tracking)
The code for this step is contained in **"./code/findingLines.py"**
```python
    def search_around_poly(fname, binary_warped, left_line, right_line, visuOn = True, writeOn = True):
```

When processing a video we get consecutive images which will not dramatically change. So we can track the lines first frame by frame and then also over multiple frames.

Frame by Frame:

 So it can be assumed that the previously detected lines should remain within a certain margin around the detected polygon. So I took the polygon from previous frame and search for all points within the warped binary image around the polygon with a margin. With those points a 2nd order polygon is fitted.

Over multiple frames:

The code for this step is contained in **"./code/findingLines.py"**
```python
    def fit_poly(line, height, ploty):
```

To overcome the issue of flickering lanes we can track the lines over multiple frame. (Some kind of smoothening) The fitted lines are pushed into the lines class and stored for n (20) frames. So then we would have for the n frames the points. Over those points I calculate again a 2nd order polygon. That is then the best fitted line. For drawing the lane on the the image out of the best fitted left and right lane a mask is created additionally:

warped             |  Sliding_window
:-------------------------:|:-------------------------:
![][warped]  |  ![][findLines]

### 5. Calculate lane curvature and ego position
The code for this step is contained in **"./code/curvature.py"**

The radius of the curvature is calculated in:
```python
    def measure_curvature_real(y_eval,left_line, right_line):
```
Therefore the given equation from the instructions is implemented to retrieve the radius for left and right line separately.
![][curvature] 

The mean of both will be displayed in later step. For conversion from pixel to real I have retrieved the height in pixel of a dashed line:

![][dashed_line] 

The relative ego position within the lane is calculated in:
```python
    def measure_position_real(left_line, right_line, img_size):
```
First the x value at the image bottom for the left and right best fit line is calculated. Then the lane center x value is calculated. With the assumption that the camera is mounted exactly in the center of the ego vehicle the image center will be also the vehicle center. So the different between lane and image center would be the vehicle offset to lane center in pixels. That is then multiplied with `xm_per_pix`. This value is gathered by looking into the pixel with of a straight line in the warped image:
![][lane_width]

#### 6. Draw results onto image
The code for this step is contained in **"./code/pipeline.py"**
```python
    def run(img, fname = None, visuOn = False, writeOn = False):
```
The findingLines returned as mask with the lane in birdyEye image. That needs to be warped back with the inverted transformation matrix which is done in:
```python
    cv2.warpPerspective(lane_mask_top_view, np.linalg.inv(M), (img_size[1], img_size[0])) 
```
Then everything is drawn onto the input image:
Original             |  Result
:-------------------------:|:-------------------------:
![][original]  |  ![][output]

---

### Pipeline (video)

The pipieline is same as above. Just that on the video we can now benefit of the tracking algorithm.

Here's a [link to my video result](./output_images/06_result/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In general all static parameters could make a problem in different situations.
1. The thresholds for the lightness and saturation will change for different lighting conditions (sunset, night etc.) A dynamic feedback  from the detected lines regarding HLS valus might help.
2. The perspective transform is done with static source and destination points. Here we highly rely on a planar and static surface. But especially at hilly roads this would fail. Here it is required to detect the surface of the road and consider it in the transformation.
3. Performing the current algorithm on the challenging video shows that differentiation between line marking as asphalt joint is not good. Here we need to differentiate if the brightness changes as "bright-dark-bright"(asphalt join) or "dark-bright-dark"(line marking)
4. Currently I am tracking the lines over 20 Frames which helps to reduce false detection on highways. But as seen in the challenging videos on rural roads the road  curvature is much more dynamic. So we cannot track the the lines like this and make them so stiff. We need a mechanism to make the tracking length dependant on the detection quality. As well we can make the margin in which new lines are searched bigger in the far distance than in the nearer distance.
5. As the lane will have tangential the same lane width. We can use to bind left and right line marking like this. That would make the lines mor parallel.

