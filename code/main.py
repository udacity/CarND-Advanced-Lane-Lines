import calibration
import pipeline
import glob
import cv2
import os

#################################################################
# Calculation on test images                                    #
#################################################################
def run_test_images():
    # Make a list of images
    img_path = './../test_images/*.jpg'
    images = glob.glob(img_path)

    print('Starting to process: ' + img_path)
    for fname in images:
        # read current image
        img = cv2.imread(fname)
        print("INPUT: " + fname)

        merged = pipeline.run(img,fname)

        cv2.imshow('merged',merged)
        cv2.waitKey(500)

        # save into file
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/06_merged' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, merged)

    cv2.destroyAllWindows()

#################################################################
# MAIN                                                          #
#################################################################
if __name__ == "__main__":
    # calibrate once
    # find the corners in the calibration images and save the output into file
    calibration.calc_calibration_parameter()
    
    run_test_images()