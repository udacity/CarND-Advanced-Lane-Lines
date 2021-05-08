import numpy as np

# Define conversions in x and y from pixels space to meters
lane_width_real = 3.7 # meters
dashed_line_length_real = 3 # meters
ym_per_pix = dashed_line_length_real/80 # meters per pixel in y dimension
xm_per_pix = lane_width_real/780 # meters per pixel in x dimension

def measure_curvature_real(y_eval,left_line, right_line):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''    
    # convert from pixel to real
    y_eval = y_eval * ym_per_pix
    
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    # left line
    delitmeter = np.absolute(2*left_fit[0])
    intern = 2*left_fit[0]*y_eval+left_fit[1]
    zaeler = (1+intern**2)**(3/2)
    left_curverad = zaeler/delitmeter
    
    # right line
    delitmeter = np.absolute(2*right_fit[0])
    intern = 2*right_fit[0]*y_eval+right_fit[1]
    zaeler = (1+intern**2)**(3/2)
    right_curverad = zaeler/delitmeter

    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad
    
    curvature_mean = np.mean([left_curverad, right_curverad])

    return curvature_mean

def measure_position_real(left_line, right_line, img_size):
    '''
    Calculates the position of the ego vehicle
    '''
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    p_left = np.poly1d(left_fit)
    leftBottom = p_left(img_size[0])

    p_right = np.poly1d(right_fit)
    rightBottom = p_right(img_size[0])

    left_line.line_base_pos = leftBottom
    right_line.line_base_pos = rightBottom


    lane_width_pixel = rightBottom - leftBottom
    lane_center_pixel = leftBottom + 0.5 * lane_width_pixel
    image_center_pixel = img_size[1] /2
    
    offset_pixel = lane_center_pixel - image_center_pixel

    #Convert from pixel to real
    # Check if dynamically the ratio shall be calculated
    # xm_per_pix_current = lane_width_real / lane_width_pixel
    
    offset_real = offset_pixel * xm_per_pix

    return offset_real
