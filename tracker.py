import cv2
import numpy as np
import time
import math
import sys


# Create a VideoCapture object and creates a mask for the subject
# in the frame. The mask is used to track the subject in the frame.
# currently limited to one subject in the frame
class Tracker:
    def __init__(self):
        # Open the default camera (usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)
        self.prev_frame = None
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            return

        # Release the camera and close all windows

    def display_camera(self):
        # Display the frame in a window named "Camera"
        frame_count = 0
        # default values for the center and radius of the subject
        centroids = [self.width / 2, self.height / 2]
        avg_radius = 100

        for ret, frame in self.frame_generator():
            # Check if the frame was successfully read
            if not ret:
                print("Failed to capture frame.")
                break

            # we need to update the center and radius of the subject every 10 frames
            # but if there is no movement in the frame, we don't need to update the center and radius
            # so check if mask_shape returns nan values
            if frame_count % 10 == 0:
                centroids1, avg_radius1 = self.mask_shape(frame)
                if not math.isnan(centroids1[0]) and not math.isnan(avg_radius1):
                    centroids = centroids1
                    avg_radius = avg_radius1
            centroids = (int(centroids[0]), int(centroids[1]))
            avg_radius = int(avg_radius)


            # the center of the subject 
            # cv2.circle(frame, (centroids[1], centroids[0]), 1, (0, 0, 255), 2)

            # circle bounding the subject
            # cv2.circle(frame, (centroids[1], centroids[0]), avg_radius, (0, 0, 255), 2)
            top_left = (centroids[1] - avg_radius, centroids[0] - avg_radius)
            bottom_right = (centroids[1] + avg_radius, centroids[0] + avg_radius)
            # rectangle bounding the subject
            cv2.rectangle(frame, top_left, bottom_right,  (0, 0, 255), 2)


            # Display the frame in a window named "Camera"
            cv2.imshow("Camera", frame)
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def frame_generator(self):
        while True:
            ret, frame = self.cap.read()
            yield ret, frame

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

        cv2.imshow("Camera2", frame_diff)

        self.prev_frame = gray_frame
        return centroids, avg_radius

    # get_avg_radius numpy loop version
    def get_avg_radius_loop(self, frame, center):
        avg_radius = 0
        for x in range(len(frame)):
            for y in range(len(frame[0])):
                if frame[x][y] > 0:
                    avg_radius += math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        avg_radius /= len(frame)
        return avg_radius

    # get_center numpy loop version
    def get_center_loop(self, frame):
        avg_x = 0
        avg_y = 0
        for x in range(len(frame)):
            for y in range(len(frame[0])):
                if frame[x][y] > 0:
                    avg_x += x
                    avg_y += y
        avg_x /= len(frame)
        avg_y /= len(frame[0])
        return avg_x, avg_y

    # np.vectorize(func) will return a function that can be applied to a numpy array
    def get_avg_radius(self, frame, center):
        x_inds, y_inds = np.nonzero(
            frame
        )  # gets the x and y indices of all the nonzero values in the frame
        y_inds = y_inds - center[1]  # subtract the center from the y values
        x_inds = x_inds - center[0]  # subtract the center from the x values
        avg_radius = np.sqrt(
            np.mean(x_inds**2 + y_inds**2)
        )  # get the average radius
        return avg_radius

    # get_avg_radius numpy verctorizer version

    def get_center(self, frame):
        return (
            np.nonzero(frame)[0].mean(),
            np.nonzero(frame)[1].mean(),
        )  # get the average x and y values of all the nonzero values in the frame

    def kalman_filter(self, centroids, velocity, acceleration):
        pass


if __name__ == "__main__":
    tracker = Tracker()
    tracker.display_camera()
