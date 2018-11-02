import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import glob



def Calibrate_Cam (img):
    #to read all the calibration images
    images = glob.glob("camera_cal/calibration*.jpg")
    nx, ny = 9,6 #number of x and y corners for the chessboards we have

    objpoints =[] #3d points for the real world points
    imgpoints =[] #2d points for the image points


    ##To Create the object points for one iteration
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    #To calculate all objpoints and all imgpoints
    for image in images:
        ret, corners = cv2.findChessboardCorners(cv2.imread(image), (nx,ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    #To calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort (img, mtx, dist, None, mtx)
    return dst

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sx_abs = np.absolute( sx )
    scaled_sx = np.uint8(255*sx_abs/np.max(sx_abs))
    binary_output = np.zeros_like(img) # Remove this line
    binary_output [ (scaled_sx>thresh[0]) & (scaled_sx<thresh[1]) ] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    s_mag = np.sqrt(sx*sx + sy*sy)
    scaled_s = np.uint8(255*s_mag/np.max(s_mag))
    binary_output = np.zeros_like(img)
    binary_output [ (scaled_s>mag_thresh[0]) & (scaled_s<mag_thresh[1]) ] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sx_abs = np.absolute(sx)
    sy_abs = np.absolute(sy)
    directions = np.arctan2(sy_abs, sx_abs)
    binary_output = np.zeros_like(img)
    binary_output [ (directions>thresh[0]) & (directions<thresh[1]) ] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S>thresh[0]) & (S<=thresh[1]) ]=1
    return binary_output

def Color_Gradient_Threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    gradx = abs_sobel_thresh(gray, orient='x',thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', thresh=(20, 100))
    grad_mag = mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
    grad_dir = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    binary [ ((gradx == 1) & (grady==1)) | ((grad_mag==1) & (grad_dir==1)) ] =1
    hls_binary = hls_select(img, thresh=(170, 255))
    color_binary = np.dstack((binary , np.zeros_like(hls_binary), hls_binary)) * 255
    return color_binary


def bird_eye (img):
    src=np.float32([])
    dst=np.float32([])
    M = cv2.getPrespectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])

    warped = cv2.warpPrespective(img,M , img_size, flags=cv2.INTER_LINEAR)
    return warped


###### Prespective Transform (Warp image)


###### Detect Line Lanes


###### Determine lane curveture


img=mpimg.imread("test_images/straight_lines1.jpg")

## Camera Calibration and image distortion
undestorted = Calibrate_Cam(img)

###### Color and Gradient Threshold
binary = Color_Gradient_Threshold(undestorted)

plt.imshow(undestorted)
plt.show()
plt.imshow(binary, cmap='gray')
