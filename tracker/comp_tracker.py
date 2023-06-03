import cv2
import numpy as np
import math

class comp_tracker:
    def __init__(self) -> None:
        self.prev_frame = None 
        self.prev_radius = [0,0]
        self.prev_centroid = [0,0]



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
        # cv2.imshow("Camera2", frame_diff)
        self.prev_frame = gray_frame
        return centroids, avg_radius


    # np.vectorize(func) will return a function that can be applied to a numpy array
    def get_avg_radius(self, frame, center):
        x_inds, y_inds = np.nonzero(
            frame
        )  # gets the x and y indices of all the nonzero values in the frame
        avg_x_offset = np.mean(np.sqrt((x_inds - center[0])**2))
        avg_y_offset = np.mean(np.sqrt((y_inds - center[1])**2))
        if math.isnan(avg_x_offset):
            avg_x_offset = self.prev_radius[0]
        if math.isnan(avg_y_offset):
            avg_y_offset = self.prev_radius[1]
        if len(np.nonzero(frame)[0])  < 10000:
            avg_x_offset = self.prev_radius[0]
            avg_y_offset = self.prev_radius[1]
        self.prev_radius = [avg_x_offset, avg_y_offset]

        return [avg_x_offset, avg_y_offset]

    # get_avg_radius numpy verctorizer version

    def get_center(self, frame):
        cent =  np.array([
            np.nonzero(frame)[1].mean(),
            np.nonzero(frame)[0].mean(),
          ])  # get the average x and y values of all the nonzero values in the frame
        
        if math.isnan(cent[0]):
            cent[0] = self.prev_centroid[0]
        if math.isnan(cent[1]):
            cent[1] = self.prev_centroid[1]
        if len(np.nonzero(frame)[0])  < 5000:
            cent = self.prev_centroid
        self.prev_centroid = cent 
        return cent
