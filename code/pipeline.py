import distortion
import transform
import glob
import cv2
import binary
import numpy as np
import findingLines
import curvature
import os

#################################################################
# TEST CALIB IMG: undistortion and transformation on calibration images    #
#################################################################
def test_undist_and_transform():
    path = '../camera_cal/calibration*.jpg'

    print('Starting to undistortion over: ' + path)
    images = glob.glob(path)
    for fname in images:
        # read current image
        img = cv2.imread(fname)
        print("INPUT: " + fname)
        
        # undistort the calibration images
        undistorted= distortion.undistort_image(fname, img)

        # transform the calibration images
        transform.corners_unwarp(fname, undistorted)

    cv2.destroyAllWindows()

#################################################################
# TEST ROAD IMG: full pipeline on single images                 #
#################################################################
def run(img, fname, visuOn = True, writeOn = True):
    img_size = img.shape

    # 1) undistort the image
    undistorted = distortion.undistort_image(fname, img, visuOn, writeOn)

    # 2) calculate the binary image
    binary_img = binary.create_binary_and_save(fname, undistorted, visuOn, writeOn)

    # create symmetric trapezoid as source
    topLeft = [590,450]
    bottemLeft = [180,720]
    topRight = [690,450]
    bottomRight = [1100,720]
    src = np.float32([topLeft,topRight,bottemLeft,bottomRight])

    # create vertical lines in image (birdsEye)
    dst = np.float32([[250, 100],   #topLeft
                    [1030, 100],     #topRight
                    [250, 720],     #bottomLeft
                    [1030, 720]])    #bottomRight

    # 3) transform to birdsEye
    [warped, M] = transform.transform_and_warp_and_save(fname, binary_img, src, dst, visuOn, writeOn)
    #transform.transform_and_warp_and_save(fname, img, src, dst, visuOn, writeOn, '_img')

    # 4) find lines
    # sliding window shall be used if no lines are detected previously
    [left_fit, right_fit, lane_mask_top_view] = findingLines.sliding_window(fname, warped, visuOn, writeOn)

    # 5) calc curvature and position
    curvature_lane = curvature.measure_curvature_real(img_size[0], left_fit, right_fit)
    position = curvature.measure_position_real(left_fit, right_fit, img_size)

    # 6) draw lane and data to the image
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    lane_mask_img = cv2.warpPerspective(lane_mask_top_view, np.linalg.inv(M), (img_size[1], img_size[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, lane_mask_img, 0.3, 0)

    radiusText = 'Radius of Curvature = ' + str(int(curvature_lane)) + '(m)'
    cv2.putText(result,radiusText,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if position < 0:
        positionText = 'Vehicle is ' + str(abs(round(position,2))) + '(m) right of center'
    else:
        positionText = 'Vehicle is ' + str(abs(round(position,2))) + '(m) left of center'
    cv2.putText(result,positionText,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result