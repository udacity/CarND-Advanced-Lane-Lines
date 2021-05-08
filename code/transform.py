import cv2
import numpy as np
import os

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# create symmetric trapezoid as source
src = np.float32([[590,450],         #topLeft
                [690,450],          #topRight
                [180,720],          #bottomLeft
                [1100,720]])        #bottomRight

# create vertical lines in image (birdsEye)
dst = np.float32([[250, 100],       #topLeft
                [1030, 100],        #topRight
                [250, 720],         #bottomLeft
                [1030, 720]])       #bottomRight

def transform_and_warp(undist, src, dst):
    # a) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # b) use cv2.warpPerspective() to warp image to a top-down view
    sizes = (undist.shape[1], undist.shape[0])
    warped = cv2.warpPerspective(undist, M, sizes, flags=cv2.INTER_LINEAR)
    
    return [warped, M]


def transform_and_warp_and_save(fname, undist, visuOn = True, writeOn = True, name=None):
    [warped, M] = transform_and_warp(undist,src, dst)
        
    # Display
    if visuOn:
        if name is not None:
            cv2.imshow('tranform',warped)
        else:
            cv2.imshow(name,warped)
        cv2.waitKey(500)

    # save into file
    if writeOn:
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        if name is not None:
            fileName = os.path.splitext(os.path.basename(fname))[0] + name +'.png'
        else:
            fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/04_transform' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, warped)

    return [warped, M]

def corners_unwarp(fname, undist):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
    # 2) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 3) If corners found: 
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

        transform_and_warp_and_save(fname, undist, src, dst)
        
    else:
        # If not all checkboard corners are seen in the image this step failes
        print ('finding corners failed for:' + fname)