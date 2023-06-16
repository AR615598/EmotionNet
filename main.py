import cv2
import numpy as np
import math
import sys
import logging
from trackers import boring_tracker as boring
from trackers import NN_tracker as nn
from trackers import contour_tracker as cont
from trackers import comp_tracker as comp
from classifiers import classifier as classifier
import main
import argparse
import threading
import os
from pynput import keyboard


# Create a VideoCapture object and creates a mask for the subject
# in the frame. The mask is used to track the subject in the frame.
# currently limited to one subject in the frame
class EmotionNet:
    def __init__(self, tracker_type):

        logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

        # Open the default camera (usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)
        self.tracker_type = tracker_type
        self.classifier = classifier.classifier()
        self.track = self.init_tracker()
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.centroids = [self.width / 2, self.height / 2]
        self.avg_radius = 1
        self.current_mask = [None, None] # [center, radius]
        self.box_data = []

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
    def change_tracker(self, tracker_type):
        self.tracker_type = tracker_type
        self.track = self.init_tracker()
    def frame_generator(self):

        while True:
            ret, frame = self.cap.read()
            yield ret, frame
    def crop_frame(self, frame, centroids, avg_radius):
        top_left = [centroids[0] - avg_radius[0], centroids[1] + avg_radius[1]]
        bottom_right = [centroids[0] + avg_radius[0], centroids[1] - avg_radius[1]]


        # check if it is within the bounds of the frame
        # first check if the top left x is less than 0 and y is greater than the height
        if top_left[0] < 0:
            top_left[0] = 0
        if top_left[1] > self.height:
            top_left[1] = self.height
        # check if the bottom right x is greater than the width and y is less than 0
        if bottom_right[0] > self.width:
            bottom_right[0] = self.width
        if bottom_right[1] < 0:
            bottom_right[1] = 0
        top_left = [int(top_left[0]), int(top_left[1])]
        bottom_right = [int(bottom_right[0]), int(bottom_right[1])] 
        
        # now we can crop the frame
        cropped_frame = frame[bottom_right[1]:top_left[1], top_left[0]:bottom_right[0]]
        # but we need to resize it to 48x48
        cropped_frame = cv2.resize(cropped_frame, (48, 48)) 
        return cropped_frame
    def frame_to_png(self, frame):
        # needs to be visaible to the classifier
        path = "classifiers/frame.jpg"
        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # check if the file was written
        if not os.path.exists(path):
            exit(1)

        return path
    def classify_emotion(self, frame):
        # classify the emotion
        emotion = self.classifier(frame)
        return emotion
    def find_mask(self, frame):
        centroids, avg_radius = self.track.mask_shape(frame)
        # we need to average the center and radius over a few frames
        # to get a more accurate representation of the subject
        if len(self.box_data) < 10:
            # if it is less than 4 frames, we just append the data
            self.box_data.append([centroids, avg_radius])
        if len(self.box_data) == 10:
            # if it is equal to 4 frames, assume the list is full
            # and we can pop the first element and append the new data
            self.box_data.pop(0)
            self.box_data.append([centroids, avg_radius])
    def average_mask(self, frame):
        averages = np.mean(self.box_data, axis=0)
        centroids = averages[0]
        avg_radius = averages[1]
        return centroids, avg_radius
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
    def draw_emotion(self, frame, emotion, topL):
        # draw the emotion above the bounding box
        cv2.putText(frame, emotion, topL, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    def main(self):
        # Display the frame in a window named "Camera"
        frame_count = 0
        camera_flag = True
        mask_flag = True
        classification_flag = True
        emotion_flag =  True
        emotion = None
        
        # captures keypresses
        def on_press(key):
            nonlocal camera_flag, mask_flag, emotion_flag, classification_flag
            # Switching trackers
            if key == keyboard.KeyCode.from_char('1'):
                self.change_tracker('comp')
            if key == keyboard.KeyCode.from_char('2'):
                self.change_tracker('cont')
            if key == keyboard.KeyCode.from_char('3'):
                self.change_tracker('NN')
            if key == keyboard.KeyCode.from_char('4'):
                self.change_tracker('boring')
            
            # Other flags
            if key in (keyboard.KeyCode.from_char('q'), keyboard.Key.esc):
                return False
                # Exit the loop
            if key == keyboard.KeyCode.from_char('m'):
                mask_flag = not mask_flag
            if key == keyboard.KeyCode.from_char('c'):
                camera_flag = not camera_flag
            if key == keyboard.KeyCode.from_char('e'):
                emotion_flag = not emotion_flag
            
        listener = keyboard.Listener(on_press=on_press)
        listener.start()



        for ret, frame in self.frame_generator():
            # Check if the frame was successfully read
            if not ret:
                print("Failed to capture frame.")
                break 
                
            # if mask flag is true, we display the mask
            # but we need to find the mask first
            if mask_flag:
                self.find_mask(frame)

                # every 4 frames, we draw the bounding box 
                if  frame_count != 0 and len(self.box_data) == 10:
                    # we need to average the center and radius over a few frames
                    # to get a more accurate representation of the subject
                    # box_data = [[center, radius],[center, radius],[center, radius],[center, radius]]
                    centroids, avg_radius = self.average_mask(frame)
                    self.current_mask = [centroids, avg_radius]

  
                if self.current_mask[0] is not None and self.current_mask[1] is not None:    
                    self.draw_bounds(frame, self.current_mask[0], self.current_mask[1])
                    if emotion_flag:
                        topL =  (self.centroids[0] - self.avg_radius[0], self.centroids[1] - self.avg_radius[1] - 15)
                        self.draw_emotion(frame, emotion, topL)

                
                    # if we have the average center and radius, we can draw the bounding box
                    # and with that we can crop the frame and classify the emotion
                    # we will do this every 10 frames
                    if frame_count % 10 == 0:
                        # if emotion flag is true, we display the emotion
                        # but we need to classify the emotion first

                        if emotion_flag:    
                            self.crop_frame(frame, centroids, avg_radius)
                            path = self.crop_frame(frame, centroids, avg_radius)
                            path = self.frame_to_png(path)
                            emotion = self.classify_emotion(path)
                    
      

            # if it is true, we display the camera
            if camera_flag:
                cv2.imshow(self.tracker_type, frame)
            frame_count += 1
            # needs to wait unless the frame will not be displayed
            cv2.waitKey(1)
            # if the listener detects a keypress, we exit the program
            if not listener.running:
                break

        cv2.destroyAllWindows()    
        self.cap.release()

   

def main():
    # the default tracker is the comparison tracker
    emotionNet = EmotionNet("comp")
    emotionNet.main()

if __name__ == "__main__":
    main()



