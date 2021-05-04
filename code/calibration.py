import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle
import os

def calc_calibration_parameter():

    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')
    print('Starting to find corners in calibration*.jpg')
    for fname in images:
        # read current image
        img = cv2.imread(fname)
        print("INPUT: " + fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found,, add object points, image points and draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

            # Display
            cv2.imshow('img',img)
            cv2.waitKey(500)

            # save into file
            fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
            outputFilePath = os.path.join('./../output_images/01_calibration' ,fileName)
            outputFilePath = os.path.normpath(outputFilePath)
            print("OPUTPUT: " + outputFilePath)
            cv2.imwrite(outputFilePath, img)
        else:
            # If not all checkboard corners are seen in the image this step failes
            print ('finding corners failed for:' + fname)
            

    cv2.destroyAllWindows()

    # save the points into a file
    data = { "objpoints" : objpoints, "imgpoints" : imgpoints}
    pickle.dump( data, open( "wide_points_pickle.p", "wb" ) )

    # save the camera matrix and distortion coefficients
    img = cv2.imread(images[0])
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    data = { "mtx" : mtx, "dist" : dist}
    pickle.dump( data, open( "wide_dist_pickle.p", "wb" ) )

