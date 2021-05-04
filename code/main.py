import calibration
import distortion
import transform
import glob
import cv2
import binary
import numpy as np
import findingLines
import curvature
import os

# find the corners in the calibration images and save the output into file
# execution is once enough
#calibration.calc_calibration_parameter()

#################################################################
# TEST undistortion and transformation on calibration images    #
#################################################################

# using the points apply undistortion onto the calib images
#distortion.undistort_image_folder('../camera_cal/calibration*.jpg')

# transform the calibration images
#transform.corners_unwarp()


#################################################################
# Calculation on test images                                    #
#################################################################
# Make a list of images
img_path = './../test_images/*.jpg'
images = glob.glob(img_path)

print('Starting to process: ' + img_path)
for fname in images:
    # read current image
    img = cv2.imread(fname)
    print("INPUT: " + fname)

    # undistort the image
    undistorted = distortion.undistort_and_save(fname, img)

    # calculate the binary image
    binary_img = binary.create_binary_and_save(fname, img)

    # create symmetric trapezoid as source
    topLeft = [590,450]
    bottemLeft = [180,720]
    topRight = [690,450]
    bottomRight = [1100,720]
    src = np.float32([topLeft,topRight,bottemLeft,bottomRight])

    # create vertical lines in image (birdsEye)
    img_size = binary_img.shape
    dst = np.float32([[300, 100],   #topLeft
                    [980, 100],     #topRight
                    [300, 720],     #bottomLeft
                    [980, 720]])    #bottomRight

    # transform to birdsEye
    [warped, M] = transform.tranform_and_warp_and_save(fname, binary_img, src, dst)
    #transform.tranform_and_warp_and_save(fname, img, src, dst, '_img')

    [ploty, left_fit, right_fit, mask, left_fitx, right_fitx] = findingLines.sliding_window(fname, warped)
    #out_img = fit_polynomial(binary_warped)

    left_curverad, right_curverad = curvature.measure_curvature_real(ploty, left_fit, right_fit)
    curvature_mean = np.mean([left_curverad, right_curverad])
    print(left_curverad, right_curverad)

    position = curvature.measure_position_real(left_fit, right_fit, img_size)
    print(position)


    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    cv2.polylines(warp_zero, np.int_([pts_left]), False, (255, 0, 0),15)
    cv2.polylines(warp_zero, np.int_([pts_right]), False, (0, 0, 255),15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, np.linalg.inv(M), (img_size[1], img_size[0])) 
    # Combine the result with the original image
    merged = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    radiusText = 'Radius of Curvature = ' + str(int(curvature_mean)) + '(m)'
    cv2.putText(merged,radiusText,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if position < 0:
        positionText = 'Vehicle is ' + str(abs(round(position,2))) + '(m) right of center'
    else:
        positionText = 'Vehicle is ' + str(abs(round(position,2))) + '(m) left of center'
    cv2.putText(merged,positionText,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('merged',merged)
    cv2.waitKey(500)

    # save into file
    fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
    outputFilePath = os.path.join('./../output_images/06_merged' ,fileName)
    outputFilePath = os.path.normpath(outputFilePath)
    print("OPUTPUT: " + outputFilePath)
    cv2.imwrite(outputFilePath, merged)

cv2.destroyAllWindows()