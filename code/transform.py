import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

def tranform_and_warp(undist,src, dst):
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    sizes = (undist.shape[1], undist.shape[0])
    return cv2.warpPerspective(undist, M, sizes, flags=cv2.INTER_LINEAR)

def corners_unwarp():
    # Make a list of images
    images = glob.glob('./../output_images/distortion/calibration*.png')

    print('Starting to tranform the calibration*.jpg')
    for fname in images:
        # read current image
        undist = cv2.imread(fname)
        print("INPUT: " + fname)

        # 2) Convert to grayscale
        gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
        # 3) Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        #print(corners)
        # 4) If corners found: 
        if ret == True:
            # a) draw corners
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
            minMin = corners[0]
            minMax = corners[nx-1]
            maxMin = corners[-nx]
            maxMax = corners[-1]

            src = np.float32([minMin,minMax,maxMin,maxMax])
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            img_size = undist.shape
            dst = np.float32([[img_size[1]*0.1, img_size[0]*0.1],
                            [img_size[1]*0.9, img_size[0]*0.1],
                            [img_size[1]*0.1, img_size[0]*0.9],
                            [img_size[1]*0.9, img_size[0]*0.9]])

            warped = tranform_and_warp(undist,src, dst)
                
            # Display
            cv2.imshow('undistorted',warped)
            cv2.waitKey(500)

            # save into file
            fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
            outputFilePath = os.path.join('./../output_images/transform' ,fileName)
            outputFilePath = os.path.normpath(outputFilePath)
            print("OPUTPUT: " + outputFilePath)
            cv2.imwrite(outputFilePath, warped)
        else:
            # If not all checkboard corners are seen in the image this step failes
            print ('finding corners failed for:' + fname)