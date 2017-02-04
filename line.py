import numpy as np
import utils

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def process_image(self, img, center):
        if not self.detected is None:
            self.ally, self.allx = utils.get_lane(img, center)
        else:
            self.ally, self.allx = utils.find_lane(img, self.current_fit)

        self.current_fit = utils.get_line_coef(self.allx, self.ally)
        self.detected = True

        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        x = utils.fit_line(self.current_fit, y)
        self.radius_of_curvature = utils.get_curvature(x, y)

