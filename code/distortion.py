import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def undistort_image(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def undistort_and_save(fname, img):
    undistorted = undistort_image(img)

    # Display
    cv2.imshow('undistorted',undistorted)
    cv2.waitKey(500)

    # save into file
    fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
    outputFilePath = os.path.join('./../output_images/02_distortion' ,fileName)
    outputFilePath = os.path.normpath(outputFilePath)
    print("OPUTPUT: " + outputFilePath)
    cv2.imwrite(outputFilePath, undistorted)

    return undistorted

def undistort_image_folder(path):
    # Make a list of calibration images
    images = glob.glob(path)

    print('Starting to undistortion over: ' + path)
    for fname in images:
        # read current image
        img = cv2.imread(fname)
        print("INPUT: " + fname)
        
        undistorted= undistort_and_save(fname, img)

    cv2.destroyAllWindows()