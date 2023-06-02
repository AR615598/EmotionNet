import cv2
import numpy as np
import math

class comp_tracker:
    def __init__(self, cap) -> None:
        self.prev_frame = None
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    



    def mask_shape(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray_frame
        # this is the difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(self.prev_frame, gray_frame)

        # 0, 255 scaling is used to make the white outline of the subject
        frame_diff = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)[1]

        centroids = self.get_center(frame_diff)
        avg_radius = self.get_avg_radius(frame_diff, centroids)
        self.prev_frame = gray_frame
        return centroids, avg_radius


    # np.vectorize(func) will return a function that can be applied to a numpy array
    def get_avg_radius(self, frame, center):
        x_inds, y_inds = np.nonzero(
            frame
        )  # gets the x and y indices of all the nonzero values in the frame
        y_inds = y_inds - center[0]  # subtract the center from the y values
        x_inds = x_inds - center[1]  # subtract the center from the x values
        avg_radius = np.sqrt(
            np.mean(x_inds**2 + y_inds**2)
        )  # get the average radius
        return avg_radius

    # get_avg_radius numpy verctorizer version

    def get_center(self, frame):
        return (
            np.nonzero(frame)[1].mean(),
            np.nonzero(frame)[0].mean(),
        )  # get the average x and y values of all the nonzero values in the frame
