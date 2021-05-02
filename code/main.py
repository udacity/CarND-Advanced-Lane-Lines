import calibration
import distortion
import transform
import glob
import cv2
import binary

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

    undistorted = distortion.undistort_and_save(fname, img)

    binary.create_binary_and_save(fname, img)
    #    tranform_and_warp(undist,src, dst)

cv2.destroyAllWindows()