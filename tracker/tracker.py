import cv2
import numpy as np
import math
import sys
import logging
import comp_tracker as comp
import contour_tracker as cont
import NN_tracker as nn
import boring_tracker as boring
import argparse
import threading



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
        self.current_mask = [None, None] # [center, radius]

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Failed to open the camera.")
            return

        # Release the camera and close all windows
    def init_tracker(self):
        if self.tracker_type == 'comp':
            return comp.comp_tracker()
        elif self.tracker_type == 'cont':
            return cont.contour_tracker(self.cap)
        elif self.tracker_type == 'NN':
            return nn.NN_tracker(self.cap)
        elif self.tracker_type == 'boring':
            return boring.boring_tracker(self.cap)
        else:
            logging.exception("No valid tracker type given")
            sys.exit(1)  # Exit the program with a non-zero status code

        

    def display_camera(self):
        # Display the frame in a window named "Camera"
        frame_count = 0
        box_data = []
        # default values for the center and radius of the subject
        centroids = [self.width / 2, self.height / 2]
        avg_radius = 100

        for ret, frame in self.frame_generator():
            # Check if the frame was successfully read
            if not ret:
                print("Failed to capture frame.")
                break

            centroids, avg_radius = self.track.mask_shape(frame)
            # we need to average the center and radius over a few frames
            # to get a more accurate representation of the subject
            if len(box_data) < 10:
                # if it is less than 4 frames, we just append the data
                box_data.append([centroids, avg_radius])
            if len(box_data) == 10:
                # if it is equal to 4 frames, assume the list is full
                # and we can pop the first element and append the new data
                box_data.pop(0)
                box_data.append([centroids, avg_radius])

            # every 4 frames, we draw the bounding box 
            if frame_count % 1 == 0 and frame_count != 0:
                # we need to average the center and radius over a few frames
                # to get a more accurate representation of the subject
                # box_data = [[center, radius],[center, radius],[center, radius],[center, radius]]
                averages = np.mean(box_data, axis=0)
                centroids = averages[0]
                avg_radius = averages[1]
                self.current_mask = [centroids, avg_radius]
            if self.current_mask[0] is not None and self.current_mask[1] is not None:    
                self.draw_bounds(frame, self.current_mask[0], self.current_mask[1])


            frame_count += 1
            # Display the frame in a window named "Camera"
            cv2.imshow(self.tracker_type, frame)
            if cv2.getWindowProperty(self.tracker_type, cv2.WND_PROP_VISIBLE) < 1:
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
    def close_all_windows(self):
        while True:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                cv2.destroyAllWindows()



    def draw_bounds(self, frame, centroids, avg_radius):
        if not math.isnan(centroids[0]) and not math.isnan(centroids[1]) and not math.isnan(avg_radius[0]) and not math.isnan(avg_radius[1]):
            self.centroids = centroids
            self.avg_radius = avg_radius
        self.centroids = [int(self.centroids[0]), int(self.centroids[1])]
        self.avg_radius = [int(self.avg_radius[0]), int(self.avg_radius[1])]
        # the center of the subject 
        cv2.circle(frame, (self.centroids[0], self.centroids[1]), 1, (0, 0, 255), 2)

  
        top_left = (self.centroids[0] - self.avg_radius[0], self.centroids[1] + self.avg_radius[1])
        bottom_right = (self.centroids[0] + self.avg_radius[0], self.centroids[1] -self.avg_radius[1])
        # rectangle bounding the subject
        cv2.rectangle(frame, top_left, bottom_right,  (0, 0, 255), 2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize tracker')
    parser.add_argument('--type', type=str, help='type of tracker to use')
    args = parser.parse_args()
    tracker = Tracker(args.type)
    tracker.display_camera()

