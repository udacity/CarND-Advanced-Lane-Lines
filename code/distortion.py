import pickle
import cv2
import glob
import os

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def undistort_image(fname, img, visuOn = True, writeOn = True):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # Display
    if visuOn:
        cv2.imshow('undistorted',undistorted)
        cv2.waitKey(500)

    # save into file
    if writeOn:
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/02_distortion' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, undistorted)

    return undistorted