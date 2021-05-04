import numpy as np
import cv2
import os

# TODO: play with thresholds
def create_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def create_binary_and_save(fname, img, visuOn = True, writeOn = True):
    color_binary = create_binary(img)

    # Display
    if visuOn:
        cv2.imshow('color_binary',color_binary)
        cv2.waitKey(500)

    # save into file
    if writeOn:
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/03_color_binary' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, color_binary)

    return color_binary