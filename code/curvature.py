import numpy as np

# Define conversions in x and y from pixels space to meters
lane_width_real = 3.7 # meters
dashed_line_length_real = 3 # meters
ym_per_pix = dashed_line_length_real/80 # meters per pixel in y dimension
xm_per_pix = lane_width_real/680 # meters per pixel in x dimension

def measure_curvature_real(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)*ym_per_pix
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    delitmeter = np.absolute(2*left_fit[0])
    intern = 2*left_fit[0]*y_eval+left_fit[1]
    zaeler = (1+intern**2)**(3/2)
    left_curverad = zaeler/delitmeter  ## Implement the calculation of the left line here
    
    delitmeter = np.absolute(2*right_fit[0])
    intern = 2*right_fit[0]*y_eval+right_fit[1]
    zaeler = (1+intern**2)**(3/2)
    right_curverad = zaeler/delitmeter  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def measure_position_real(left_fit, right_fit, img_size):
    '''
    Calculates the position of the ego vehicle
    '''

    p_left = np.poly1d(left_fit)
    leftBottom = p_left(img_size[0])

    p_right = np.poly1d(right_fit)
    rightBottom = p_right(img_size[0])


    lane_width_pixel = rightBottom - leftBottom
    lane_center_pixel = leftBottom + 0.5 * lane_width_pixel
    image_center_pixel = img_size[1] /2
    
    offset_pixel = lane_center_pixel - image_center_pixel

    #Convert from pixel to real
    # Check if dynamically the ratio shall be calculated
    # xm_per_pix_current = lane_width_real / lane_width_pixel
    
    offset_real = offset_pixel * xm_per_pix

    return offset_real
