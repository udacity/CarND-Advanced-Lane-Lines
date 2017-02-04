import cv2
import numpy as np
import utils
from line import Line


class LaneFinder:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.left_line = Line()
        self.right_line = Line()

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_line_centers(self, warped):
        histogram = np.sum(warped[warped.shape[0] / 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0] / 2)
        left_center = np.argmax(histogram[:midpoint])
        right_center = np.argmax(histogram[midpoint:]) + midpoint
        return left_center, right_center

    def get_transformation_source(self, img, left=0.44, right=0.56, top=0.655, bottom=1):
        h, w = img.shape[:2]
        top_left = [w * left, h * top, ]
        top_right = [w * right, h * top]
        bottom_left = [w * 0.16, h * bottom]
        bottom_right = [w * 0.86, h * bottom]
        return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)

    def get_transformation_destination(self, img):
        h, w = img.shape[:2]
        top_left = [w * 0.25, 0]
        top_right = [w * 0.75, 0]
        bottom_left = [w * 0.25, h]
        bottom_right = [w * 0.75, h]
        return np.array([top_left, bottom_left, bottom_right, top_right], np.float32)

    def convert_to_birds_eye_view(self, img):
        src = self.get_transformation_source(img)
        dst = self.get_transformation_destination(img)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


    def draw_region(self, img, region, color=[255, 0, 0]):
        img = np.copy(img).astype(np.uint8)
        left_top = tuple(region[0])
        left_bottom = tuple(region[1])
        right_bottom = tuple(region[2])
        right_top = tuple(region[3])

        cv2.line(img, left_bottom, left_top, color, 2)
        cv2.line(img, left_top, right_top, color, 2)
        cv2.line(img, right_top, right_bottom, color, 2)
        cv2.line(img, right_bottom, left_bottom, color, 2)
        return img

    @staticmethod
    def detect_white_line(image):
        l = np.array([100, 100, 200])
        u = np.array([255, 255, 255])
        detected = cv2.inRange(image, l, u)
        binary = np.zeros(image.shape[:2])
        binary[(detected > 0)] = 1

        return binary

    @staticmethod
    def detect_yellow_line(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        l = np.array([20, 100, 100])
        u = np.array([40, 255, 255])
        detected = cv2.inRange(image, l, u)
        binary = np.zeros(image.shape[:2])
        binary[(detected > 0)] = 1

        return binary

    def create_mask(self, img):
        warped = self.undistort(img)

        # Threshold x gradient
        x_sobel = utils.abs_sobel_threshold(warped, 'x', 5, thresh=(20, 170))

        yellow = self.detect_yellow_line(warped)
        white = self.detect_white_line(warped)

        combined_binary = np.zeros_like(x_sobel)
        combined_binary[(white == 1) | (yellow == 1)] = 1

        return self.convert_to_birds_eye_view(combined_binary)

    def project_on_image(self, img, left_x, right_x, y):
        color_warp = np.zeros_like(img).astype(np.uint8)

        src = self.get_transformation_source(img)
        dest = self.get_transformation_destination(img)

        Minv = cv2.getPerspectiveTransform(dest, src)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_x, y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        return result

    def get_deviation_from_center(self, img, left, right):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        middle = img.shape[1] / 2
        dev = (left + right >> 1) - middle
        return np.round(dev * xm_per_pix, 2)

    def process_image(self, img):
        warped = self.create_mask(img)

        left_center, right_center = self.get_line_centers(warped)

        self.left_line.process_image(warped, left_center)
        self.right_line.process_image(warped, right_center)

        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        result = self.project_on_image(img, utils.fit_line(self.left_line.current_fit, y),
                                       utils.fit_line(self.right_line.current_fit, y), y)

        result = self.display_mask(result, warped)
        result = self.display_birds_eye(result)

        deviation = self.get_deviation_from_center(result, left_center, right_center)

        l_text = "Left Curvature:  {} m".format(self.left_line.radius_of_curvature)
        r_text = "Right Curvature:  {} m".format(self.right_line.radius_of_curvature)
        dev_text = "Deviation from center: {} m".format(deviation)
        cv2.putText(result, l_text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(result, r_text, (50, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(result, dev_text, (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return result

    def display_birds_eye(self, img):
        mask = self.convert_to_birds_eye_view(img)
        mask = cv2.resize(mask, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
        return self.add_display(img, mask, x_offset=img.shape[1]*0.75, y_offset=10)

    def display_mask(self, img,  warped):
        mask = np.dstack((warped, warped, warped)) * 255
        mask = cv2.resize(mask, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
        return self.add_display(img, mask, x_offset=img.shape[1]*0.5, y_offset=10)

    def add_display(self, result, display, x_offset, y_offset):
        result[y_offset: y_offset + display.shape[0], x_offset: x_offset + display.shape[1]] = display
        return result
