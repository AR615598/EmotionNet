import cv2
import numpy as np
import math
import sys
import logging
import comp_tracker as comp
import contour_tracker as cont
import NN_tracker as nn


# Create a VideoCapture object and creates a mask for the subject
# in the frame. The mask is used to track the subject in the frame.
# currently limited to one subject in the frame
class Tracker:
    def __init__(self, tracker_type):
        logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

        # Open the default camera (usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)
        self.tracker_type = tracker_type
        self.track = self.init_tracker()
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.centroids = [self.width / 2, self.height / 2]
        self.avg_radius = 1

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            return

        # Release the camera and close all windows
    def init_tracker(self):
        if self.tracker_type == 'comp':
            return comp.comp_tracker(self.cap)
        elif self.tracker_type == 'cont':
            return cont.contour_tracker(self.cap)
        elif self.tracker_type == 'NN':
            return nn.NN_tracker(self.cap)
        else:
            logging.exception("No valid tracker type given")
            sys.exit(1)  # Exit the program with a non-zero status code

        

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
                centroids, avg_radius = self.track.mask_shape(frame)
                self.draw_bounds(frame, centroids, avg_radius)

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

    def draw_bounds(self, frame, centroids, avg_radius):
        if not math.isnan(centroids[0]) and not math.isnan(centroids[1]) and not math.isnan(avg_radius):
            self.centroids = centroids
            self.avg_radius = avg_radius
        self.centroids = [int(self.centroids[0]), int(self.centroids[1])]
        self.avg_radius = int(self.avg_radius)
        # the center of the subject 
        cv2.circle(frame, (self.centroids[0], self.centroids[1]), 1, (0, 0, 255), 2)

        # circle bounding the subject
        # cv2.circle(frame, (centroids[1], centroids[0]), avg_radius, (0, 0, 255), 2)
        top_left = (self.centroids[0] - self.avg_radius, self.centroids[1] - self.avg_radius)
        bottom_right = (self.centroids[0] + self.avg_radius, self.centroids[1] + self.avg_radius)
        # rectangle bounding the subject
        cv2.rectangle(frame, top_left, bottom_right,  (0, 0, 255), 2)
    

if __name__ == "__main__":
    tracker = Tracker('comp')
    tracker.display_camera()
    tracker2= Tracker('cont')
    tracker2.display_camera()
