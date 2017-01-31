import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "camera_cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Unwarp the image
def unwarp(img, mtx, dist):
    
    img_u = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY)
    
    # Example Images source and destination coordinates for image warp
    #src = np.float32([[397, 365], [573, 365], [160, 530], [840, 530]])
    #dst = np.float32([[100, 100], [700, 100], [100, 500], [700, 500]])
    
    # Test Images and Project Video source and destination coordinates for image warp
    src = np.float32([[558, 475], [757, 475], [225, 700], [1200, 700]])
    dst = np.float32([[100, 100], [1100, 100], [100, 700], [1100, 700]])
    
    # Challenge Video source and destination coordinates for image warp
    #src = np.float32([[570, 490], [758, 490], [225, 700], [1200, 700]])
    #dst = np.float32([[100, 100], [1100, 100], [100, 700], [1100, 700]])
    
    # Perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_u, M, gray.shape[::-1], flags = cv2.INTER_LINEAR)
    
    return warped, M

# Apply sobel gradient in x or y direction
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
    
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
# Not used in the final pipeline
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
    
# Define a function to threshold an image for a given range and Sobel kernel
# Not used in the final pipeline
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image based on color hue, saturation, and/or lightness (HLS space)    
def hls_select(img, min_thresh=(0, 0, 0), max_thresh=(255, 255, 255)):
    
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    
    # Create a binary image where HLS values are within predefined thresholds
    binary_output = np.zeros_like(H)
    binary_output[(H > min_thresh[0]) & (H <= max_thresh[0]) & (L > min_thresh[1]) & (L <= max_thresh[1]) & (S > min_thresh[2]) & (S <= max_thresh[2])] = 1
    
    # Return the binary image
    return binary_output

# Image processing pipeline
def process_image(img):
    # Global variables to store (x,y) location of lane lines
    global y_val_L_prev
    global x_val_L_prev
    global y_val_R_prev
    global x_val_R_prev
    
    # Yellow and white HLS thresholds
    max_thresh_y = (50, 190, 255)
    min_thresh_y = (15, 140, 100)
    max_thresh_w = (255, 255, 255)
    min_thresh_w = (0, 200, 0)
    
    # Unit conversion from pixels to meters
    ym_per_pix = 3/150 # meters per pixel in y dimension
    xm_per_pix = 3.7/850 # meteres per pixel in x dimension
    
    # Main pipeline process
    # 1. Change image to overhead perspective
    # 2. X direction sobel gradient
    # 3. Y direction sobel gradient
    # 4. HLS filter for yellow lines
    # 5. HLS filter for white lines
    # 6. Combine various filters into one binary image
    # 7. Moving histogram window to identify left and right lane lines
    # 8. Fit a polynomial to estimated left and right lane markings
    # 9. Calculate radius of curvature
    # 10. Calculate the vehicle position in the lane
    
    # 1. Change image to overhead perspective
    top_down, perspective_M = unwarp(img, mtx, dist)
    
    # Calculate inverse to warp image back to original perspective
    Minv = np.linalg.inv(perspective_M)
    
    # 2. X direction sobel gradient
    grad_x_binary = abs_sobel_thresh(top_down, orient='x', thresh_min=20, thresh_max=170)
    
    # 3. Y direction sobel gradient
    grad_y_binary = abs_sobel_thresh(top_down, orient='y', thresh_min=20, thresh_max=100)
    
    # Magnitude and direction gradients not used in final pipeline
    #mag_binary = mag_thresh(top_down, sobel_kernel=9, mag_thresh=(30, 190))
    #dir_binary = dir_threshold(top_down, sobel_kernel=15, thresh=(0.7, 1.2))
    
    # 4. HLS filter for yellow lines
    hls_binary_yellow = hls_select(top_down, min_thresh_y, max_thresh_y)
    
    # 5. HLS filter for white lines
    hls_binary_white = hls_select(top_down, min_thresh_w, max_thresh_w)
    
    # 6. Combine various filters into one binary image
    combined = np.zeros_like(grad_x_binary)
    combined[((grad_x_binary == 1) & (grad_y_binary == 1)) | (hls_binary_yellow == 1) | (hls_binary_white == 1)] = 1
      
    # 7. Moving histogram window to identify left and right lane lines
    # Each histogram consists of half of the image space, eight histograms in total
    # The max value in the histogram on the left half and right half of the image are assumed to be lane lines
    img_mask = np.zeros_like(combined)

    for i in range(9):
        histogram = np.sum(combined[combined.shape[0]/2 - i*combined.shape[0]/16:combined.shape[0]-(i*combined.shape[0]/16),:], axis=0)
        lane_L = histogram[:histogram.shape[0]/2].argmax()
        lane_R = histogram[histogram.shape[0]/2:].argmax() + histogram.shape[0]/2
        
        # If lane location is too far from adjacent lane location, assume the value is incorrect and use adjacent value instead
        if i > 0:
            if (lane_L > lane_L_prev + 150) | (lane_L < lane_L_prev - 150):
                lane_L = lane_L_prev
            if (lane_R > lane_R_prev + 150) | (lane_R < lane_R_prev - 150):
                lane_R = lane_R_prev
        # Store lane location for current histogram for comparison to adjacent histogram
        lane_L_prev = lane_L
        lane_R_prev = lane_R
        
        # Build lane mask around identified max histogram locations
        mask = np.zeros_like(histogram)
        mask[lane_L - 30 : lane_L + 30] = 1
        mask[lane_R - 30 : lane_R + 30] = 1     
        mask_array = np.tile(mask,(combined.shape[0]/2,1))
        img_mask[combined.shape[0]/2 - i*combined.shape[0]/16:combined.shape[0]-(i*combined.shape[0]/16),:] += mask_array
        img_mask[img_mask > 0] = 1
        
    # 8. Fit a polynomial to estimated left and right lanes
    # Add max histogram mask to combined filter
    combined_masked = np.zeros_like(combined)
    combined_masked[(combined == 1) & (img_mask == 1)] = 1
    
    # Define arrays for (x,y) coordinate of estimated left and right lanes
    y_val_L = []
    x_val_L = []
    y_val_R = []
    x_val_R = []
    
    # Determine if lane coordinate is part of left or right lane
    for i in range(combined_masked.shape[0]):
        for j in range(combined_masked.shape[1]):
            if combined_masked[i,j] == 1:
                if j < (combined_masked.shape[1] / 2):
                    y_val_L.append(i)
                    x_val_L.append(j)
                else:
                    y_val_R.append(i)
                    x_val_R.append(j)  

    y_val_L = np.array(y_val_L)
    x_val_L = np.array(x_val_L)
    y_val_R = np.array(y_val_R)
    x_val_R = np.array(x_val_R)
    
    # If no lane is identified, use lane from previous image
    if not y_val_L.size:
        y_val_L = y_val_L_prev
        x_val_L = x_val_L_prev
    if not x_val_L.size:
        y_val_L = y_val_L_prev
        x_val_L = x_val_L_prev
    if not y_val_R.size:
        y_val_R = y_val_R_prev
        x_val_R = x_val_R_prev
    if not x_val_R.size:
        y_val_R = y_val_R_prev
        x_val_R = x_val_R_prev
    y_val_L_prev = y_val_L
    x_val_L_prev = x_val_L
    y_val_R_prev = y_val_R
    x_val_R_prev = x_val_R
    
    # Fit 2nd order polynomials to left lane coordinates
    left_fit = np.polyfit(y_val_L, x_val_L, 2)
    # Define a top and bottom limit to help with plotting
    y_val_L = y_val_L[y_val_L > 300]
    y_val_L = np.append(300, y_val_L)
    y_val_L = np.append(y_val_L, top_down.shape[0])
    left_fitx = left_fit[0]*y_val_L**2 + left_fit[1]*y_val_L + left_fit[2]
    
    # Fit 2nd order polynomials to right lane coordinates
    right_fit = np.polyfit(y_val_R, x_val_R, 2)
    # Define a top and bottom limit to help with plotting
    y_val_R = y_val_R[y_val_R > 300]
    y_val_R = np.append(300, y_val_R)
    y_val_R = np.append(y_val_R, top_down.shape[0])
    right_fitx = right_fit[0]*y_val_R**2 + right_fit[1]*y_val_R + right_fit[2]
    
    # 9. Calculate a radius of curvature by fitting a 2nd order polynomial half way between the left and right lane polynomials
    y_val_temp = np.array([300, 450, top_down.shape[0]])
    left_fitx_temp = left_fit[0]*y_val_temp**2 + left_fit[1]*y_val_temp + left_fit[2]
    right_fitx_temp = right_fit[0]*y_val_temp**2 + right_fit[1]*y_val_temp + right_fit[2]
    mid_fitx_temp = (left_fitx_temp + right_fitx_temp) / 2
    mid_fit = np.polyfit(y_val_temp * ym_per_pix, mid_fitx_temp * xm_per_pix, 2)
    
    # Radius of curvature calc
    rad_of_curve = round((1 + (2 * mid_fit[0] * top_down.shape[0] + mid_fit[1]) ** 2) ** (3/2) / abs(2 * mid_fit[0]), 1)
    
    # 10. Calculate the vehicle position within the lane
    # 530 pixels is the location of the center of the image, warped
    center_of_lane = 530 * xm_per_pix
    # Add half of the lane width to the left lane polynomial to estimate the vehicle location
    center_offset = round((center_of_lane - (left_fitx[-1] * xm_per_pix + 3.7 / 2)), 1)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined_masked).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, y_val_L]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_val_R])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    # Write radius of curvature and lane location to image
    cv2.putText(result, 'Radius of Curvature: ' + str(rad_of_curve) + ' m', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    if center_offset >= 0:
        cv2.putText(result, 'Distance from Center of Lane: ' + str(center_offset) + ' m to the right', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    else:
        cv2.putText(result, 'Distance from Center of Lane: ' + str(-center_offset) + ' m to the left', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    # Return the final image
    return result

# Video processing
white_output = 'project_video_labeled_saturated_2.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))