import numpy as np
import cv2
import os

def find_lane_pixels(binary_warped_color):
    binary_warped = cv2.cvtColor(binary_warped_color, cv2.COLOR_BGR2GRAY)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low), \
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        inHeight = np.where(np.logical_and(nonzeroy>=win_y_low, nonzeroy<win_y_high))
        inWidthLeft = np.where(np.logical_and(nonzerox>=win_xleft_low, nonzerox<win_xleft_high))
        inWidthRight = np.where(np.logical_and(nonzerox>=win_xright_low, nonzerox<win_xright_high))
        good_left_inds = np.intersect1d(inHeight, inWidthLeft)
        good_right_inds = np.intersect1d(inHeight, inWidthRight)
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds)> minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)> minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def sliding_window(fname, warped, left_line, right_line, visuOn = True, writeOn = True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    ## Visualization ##
    # Colors in the left and right lane regions
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    cv2.polylines(warp_zero, np.int_([pts_left]), False, (255, 0, 0),15)
    cv2.polylines(warp_zero, np.int_([pts_right]), False, (0, 0, 255),15)

    if visuOn or writeOn:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        pts_left = np.stack((left_fitx, ploty), axis=1)
        pts_left = pts_left.reshape((-1,1,2))
        cv2.polylines(out_img,np.int32([pts_left]),False,(0,255,255),5)

        pts_right = np.stack((right_fitx, ploty), axis=1)
        pts_right = pts_right.reshape((-1,1,2))
        cv2.polylines(out_img,np.int32([pts_right]),False,(0,255,255),5)

    # Display
    if visuOn:
        cv2.imshow('sliding_window',out_img)
        cv2.waitKey(500)

    # save into file
    if writeOn:
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/05_sliding_window' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, out_img)

    left_line.detected = True
    right_line.detected = True

    return warp_zero


def fit_poly(left_line, right_line):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_line.current_fit = np.polyfit(left_line.ally, left_line.allx, 2)
    right_line.current_fit = np.polyfit(right_line.ally, right_line.allx, 2)
    return None

def search_around_poly(fname, binary_warped, left_line, right_line, visuOn = True, writeOn = True):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    # p_left = np.poly1d(left_line.current_fit)
    # p_right = np.poly1d(right_line.current_fit)
    
    # # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    # right_lane_inds = []
    # for idx,y in enumerate(nonzeroy):
    #     leftx_current = p_left(y)
    #     rightx_current = p_right(y)
    #     win_xleft_low = leftx_current - margin # Update this
    #     win_xleft_high = leftx_current + margin  # Update this
    #     win_xright_low = rightx_current - margin # Update this
    #     win_xright_high = rightx_current + margin  # Update this
    #     if nonzerox[idx] >= win_xleft_low and nonzerox[idx] < win_xleft_high:
    #         left_lane_inds.append(idx)
    #     if nonzerox[idx] >= win_xright_low and nonzerox[idx] < win_xright_high:
    #         right_lane_inds.append(idx)
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    left_line.allx = nonzerox[left_lane_inds]
    left_line.ally = nonzeroy[left_lane_inds] 
    right_line.allx = nonzerox[right_lane_inds]
    right_line.ally = nonzeroy[right_lane_inds]

    # Fit new polynomials
    fit_poly(left_line, right_line)
    
    ## Visualization ##
    # Generate x and y values for plotting
    img_shape = binary_warped.shape
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### Calc both polynomials using ploty, left_fit and right_fit ###
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(binary_warped)
    # Color in left and right line pixels
    # out_img = binary_warped
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    # # left margin
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
    #                         ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # # right margin
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
    #                         ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # # Draw the margin onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # lane
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0,255, 0))
    cv2.polylines(window_img, np.int_([pts_left]), False, (255, 0, 0),15)
    cv2.polylines(window_img, np.int_([pts_right]), False, (0, 0, 255),15)

    # Display
    if visuOn:
            cv2.imshow('search around poly',window_img)
            cv2.waitKey(500)

    # save into file
    if writeOn:
        fileName = os.path.splitext(os.path.basename(fname))[0] + '.png'
        outputFilePath = os.path.join('./../output_images/05_search_and_poly' ,fileName)
        outputFilePath = os.path.normpath(outputFilePath)
        print("OPUTPUT: " + outputFilePath)
        cv2.imwrite(outputFilePath, window_img)

    return window_img