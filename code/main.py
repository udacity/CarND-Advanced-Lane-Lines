import calibration
import distortion
import transform
import glob
import cv2
import binary
import numpy as np
import findingLines
import curvature

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
#img_path = './../camera_cal/calibration*.png'
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

    # create trapezoid as source
    topLeft = [595,450]
    bottemLeft = [275,680]
    topRight = [690,450]
    bottomRight = [1045,680]
    src = np.float32([topLeft,topRight,bottemLeft,bottomRight])

    # create vertical lines in image (birdsEye)
    img_size = binary_img.shape
    dst = np.float32([[img_size[1]*0.25, img_size[0]*0],   #topLeft
                    [img_size[1]*0.75, img_size[0]*0],     #topRight
                    [img_size[1]*0.25, img_size[0]*1],     #bottomLeft
                    [img_size[1]*0.75, img_size[0]*1]])    #bottomRight

    # transform to birdsEye
    warped = transform.tranform_and_warp_and_save(fname, binary_img, src, dst)
    #transform.tranform_and_warp_and_save(fname, img, src, dst, '_img')

    [ploty, left_fit, right_fit, mask] = findingLines.sliding_window(fname, warped)
    #out_img = fit_polynomial(binary_warped)

    left_curverad, right_curverad = curvature.measure_curvature_real(ploty, left_fit, right_fit)
    curvature_mean = np.mean([left_curverad, right_curverad])
    print(left_curverad, right_curverad)


    warped_back = transform.tranform_and_warp_and_save(fname, mask, dst, src,'mask')

    # TODO - merge so that the color is ramaining
    merged = cv2.addWeighted(img,1.0,warped_back,0.5,0)
    radiusText = 'Radius of Curvature = ' + str(int(curvature_mean)) + '(m)'
    cv2.putText(merged,radiusText,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('merged',merged)
    cv2.waitKey(500)


    # add curvature and lane position

cv2.destroyAllWindows()