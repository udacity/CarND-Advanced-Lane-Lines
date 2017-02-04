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


def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_transformation_source(h, w, left=0.465, right=0.535, top=0.625, bottom=1):
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


def get_line_centers(warped):
    histogram = np.sum(warped[warped.shape[0] / 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0] / 2)
    left_center = np.argmax(histogram[:midpoint])
    right_center = np.argmax(histogram[midpoint:]) + midpoint

    return left_center, right_center


def get_lane(image, center, num_windows=9, width=100, minpix=100):
    h = image.shape[0]
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    lane = []
    window_height = h // num_windows
    current = center

    for window in range(num_windows):
        lo = h - (window + 1) * window_height
        hi = h - window * window_height
        left = current - width
        right = current + width

        good_idx = ((nonzeroy >= lo) & (nonzeroy < hi) &
                    (nonzerox >= left) & (nonzerox < right)).nonzero()[0]
        lane.append(good_idx)
        if len(good_idx) > minpix:
            current = np.int(np.mean(nonzerox[good_idx]))

    lane = np.concatenate(lane)
    return nonzeroy[lane], nonzerox[lane]


def fit_line(fit, y):
    return fit[0] * y ** 2 + fit[1] * y + fit[2]


def get_line_coef(x, y):
    return np.polyfit(y, x, 2)


def find_lane(image, fit, width=100):
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - width)) &
                      (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + width)))

    x = nonzerox[left_lane_inds]
    y = nonzeroy[left_lane_inds]
    return x, y


def get_curvature(x, y):
    y_eval = np.max(y)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit_cr[0])
    return np.round(left_curverad, 2)