import calibration
import pipeline
import glob
import cv2
import os
from moviepy.editor import VideoFileClip
import numpy as np

#################################################################
# Calculation on test images                                    #
#################################################################
def run_test_images(visuOn = True, writeOn = True):
    # Make a list of images
    #img_path = './../test_images/*.jpg'
    img_path = './../project_video_images/*.png'
    images = glob.glob(img_path)

    print('Starting to process: ' + img_path)
    for fname in images:
        # read current image
        img = cv2.imread(fname)
        print("INPUT: " + fname)

        merged = pipeline.run(img,fname, visuOn, writeOn)

        cv2.imshow('merged',merged)
        cv2.waitKey(500)

        # save into file
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/06_result' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, merged)

    cv2.destroyAllWindows()

#################################################################
# Calculation on video                                          #
#################################################################
def run_video(visuOn = False, writeOn = False):
    fileName = 'project_video.mp4'
    #fileName = 'challenge_video.mp4'
    #fileName = 'harder_challenge_video.mp4'
    video_path = os.path.join('./../', fileName)

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip1 = VideoFileClip(video_path).subclip(19,25)
    clip1 = VideoFileClip(video_path)

    # dump out few images of the video files for analysis purpose
    # timeline = np.linspace(0,int(clip1.duration),int(clip1.duration)+1)
    # for frame in timeline:
    #    if frame % 1 == 0:
    #        fileNameFrame = os.path.splitext(os.path.basename(fileName))[0] + '_' + str(int(frame)) + '.png'
    #        outputFilePath = os.path.join('./../project_video_images', fileNameFrame)
    #        clip1.save_frame(outputFilePath,frame)

    white_clip = clip1.fl_image(pipeline.run) #NOTE: this function expects color images!!

    # save into file
    outputFilePath = os.path.join('./../output_images/06_result', fileName)
    if os.path.exists(outputFilePath):
        os. remove(outputFilePath)
    white_clip.write_videofile(outputFilePath, audio=False)

#################################################################
# MAIN                                                          #
#################################################################
if __name__ == "__main__":
    # calibrate once
    # find the corners in the calibration images and save the output into file
    #calibration.calc_calibration_parameter(False, False)
    
    #pipeline.test_undist_and_transform()
    
    #run_test_images(True, True)

    run_video()