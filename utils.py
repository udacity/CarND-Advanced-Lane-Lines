import numpy as np
import cv2

def calibrate_camera(image_paths):
    if not image_paths:
        raise Exception('No image paths defined')

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    gray = None
    # Step through the list and search for chessboard corners
    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def apply_threshold(mask, thresh):
    binary = np.zeros_like(mask)
    binary[(mask >= thresh[0]) & (mask <= thresh[1])] = 1
    return binary


def get_scaled_sobel(img, orient='x', sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)

    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return sobel


def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=[0, 255]):
    sobel = get_scaled_sobel(img, orient, sobel_kernel)
    return apply_threshold(sobel, thresh)



def mag_thresh(img, sobel_kernel, thresh=[0, 255]):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)

    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(255 * sobel / np.max(sobel))

    return apply_threshold(sobel, thresh)


def dir_thresh(img, sobel_kernel, thresh=[0, np.pi / 2]):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)

    abs_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return apply_threshold(abs_dir, thresh)


def color_thresh(img, s_thresh=(170, 255)):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    s_binary = apply_threshold(s_channel, s_thresh)
    return s_binary


def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def get_transformation_source(h, w, left=0.447, right=0.555, top=0.65, bottom=1):
    top_left = [w * left, h * top, ]
    top_right = [w * right, h * top]
    bottom_left = [w * 0.16, h * bottom]
    bottom_right = [w * 0.86, h * bottom]
    return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)


def get_transformation_destination(h, w):
    top_left = [w * 0.25, 0]
    top_right = [w * 0.75, 0]
    bottom_left = [w * 0.25, h]
    bottom_right = [w * 0.75, h]
    return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` should be a blank image (all black).

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)


def convert_to_birds_eye_view(img):
    h, w = img.shape[:2]

    src = get_transformation_source(h, w)
    dest = get_transformation_destination(h, w)

    img = perspective_transform(img, src, dest)

    return img, src, dest


def color_threshold(img, channel, thresh=(0, 255)):
    if channel == 'R':
        color_mask = img[:, :, 0]
    elif channel == 'G':
        color_mask = img[:, :, 1]
    elif channel == 'B':
        color_mask = img[:, :, 2]
    elif channel == 'H':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_mask = hls[:, :, 0]
    elif channel == 'L':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_mask = hls[:, :, 1]
    elif channel == 'S':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        color_mask = hls[:, :, 2]
    else:
        color_mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return apply_threshold(color_mask, thresh)