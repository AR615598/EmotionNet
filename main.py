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
import logging
import os
import argparse




# Create a VideoCapture object and creates a mask for the subject
# in the frame. The mask is used to track the subject in the frame.
# currently limited to one subject in the frame
class EmotionNet:
    def __init__(self, tracker_type):
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
        


    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()



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



    def crop_frame(self, frame, centroids, avg_radius, size=256):
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
        cropped_frame = cv2.resize(cropped_frame, (size, size)) 
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
        centroids, avg_radius = self.verify_mask(centroids, avg_radius)
        return centroids, avg_radius
    
    def verify_mask(self, centroids, avg_radius):

        # make sure it is winthin the bounds of the frame

        # centroids[0] is the x coordinate of the center while centroids[1] is the y coordinate of the center
        # avg_radius[0] is the distance between the center and the left wall of the bounding box
        topL = [centroids[0] - avg_radius[0], centroids[1] - avg_radius[1]]
        botR = [centroids[0] + avg_radius[0], centroids[1] + avg_radius[1]]
        # check if the top left x is less than 0 and y is greater than the height
        # if it is we adjust the avg_radius accordingly
        if topL[0] < 0: 
            avg_radius[0] = avg_radius[0] + topL[0]
        if topL[1] < 0:
            avg_radius[1] = avg_radius[1] + topL[1] 
        # check if the bottom right x is greater than the width and y is less than 0
        if botR[0] > self.width:
            avg_radius[0] = avg_radius[0] + (self.width - botR[0] )
        if botR[1] > self.height:
            avg_radius[1] = avg_radius[1] + (self.height - botR[1] )
        return centroids, avg_radius






    def draw_bounds(self, frame, centroids, avg_radius):
        if not math.isnan(centroids[0]) and not math.isnan(centroids[1]) and not math.isnan(avg_radius[0]) and not math.isnan(avg_radius[1]):
            self.centroids = centroids
            self.avg_radius = avg_radius
        self.centroids = [int(self.centroids[0]), int(self.centroids[1])]
        self.avg_radius = [int(self.avg_radius[0]), int(self.avg_radius[1])]

  
        top_left = (self.centroids[0] - self.avg_radius[0], self.centroids[1] + self.avg_radius[1])
        bottom_right = (self.centroids[0] + self.avg_radius[0], self.centroids[1] -self.avg_radius[1])
        # rectangle bounding the subject
        cv2.rectangle(frame, top_left, bottom_right,  (0, 0, 255), 5)


    # Draw the text within the bounding box, make the text white in a solid red block 
    def draw_emotion(self, frame, emotion, botL):
        pad = 15
        textPad = pad/2
        # draw the red block
        # to do this, we need to know the width and height of the text
        ((x,y), _ ) = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        # x and y is the height and width of the text, so given the botL and textSize, we can find the top right
        (x2, y2) = botL
        # shifts to right 
        boxTopR = (x2 + x + pad, y2 - y)
        #shifts down 
        boxBotL = (x2, y2 + pad)
        # draw the red block
        cv2.rectangle(frame, boxTopR, boxBotL, (0, 0, 255), -1)
        # moves the bot L to half of the padding to be centered
        textBotL = (int(x2 + textPad), int(y2 - textPad + pad))
                # draw the text
        cv2.putText(frame, emotion, textBotL, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    def run(self, emotion_flag, mask_flag, camera_flag, resolution=256):
        # Display the frame in a window named "Camera"
        frame_count = 0
        emotion = None
        conf = None

        for ret, frame in self.frame_generator():
            # Check if the frame was successfully read
            if not ret:
                print("Failed to capture frame.")
                break 
                
            # if mask flag is true, we display the mask
            # but we need to find the mask first
            self.find_mask(frame)

            # every 4 frames, we draw the bounding box 
            if  frame_count != 0 and len(self.box_data) == 10:
                # we need to average the center and radius over a few frames
                # to get a more accurate representation of the subject
                # box_data = [[center, radius],[center, radius],[center, radius],[center, radius]]
                centroids, avg_radius = self.average_mask(frame)
                self.current_mask = [centroids, avg_radius]

  
            if self.current_mask[0] is not None and self.current_mask[1] is not None: 
                # if we have the average center and radius, we can draw the bounding box
                # and with that we can crop the frame and classify the emotion
                # we will do this every 10 frames
                if frame_count % 10 == 0:
                    # if emotion flag is true, we display the emotion
                    # but we need to classify the emotion first
                    self.crop_frame(frame, centroids, avg_radius)
                    path = self.crop_frame(frame, centroids, avg_radius, resolution)
                    path = self.frame_to_png(path)
                    if emotion_flag:
                        emotion, conf = self.classify_emotion(path)
                if mask_flag:   
                    self.draw_bounds(frame, self.current_mask[0], self.current_mask[1])
                    if emotion_flag and emotion is not None and conf is not None:
                        topL =  (self.centroids[0] - self.avg_radius[0], self.centroids[1] - self.avg_radius[1] + 20)
                        self.draw_emotion(frame, emotion, topL)
                        # add padding 
                        print(f"Emotion: {emotion:{6}} | Confidence: {conf}", end="\r")
                elif emotion_flag and emotion is not None and conf is not None:
                    self.draw_emotion(frame, emotion, (0,20))

            
                    
      

            # if it is true, we display the camera
            if camera_flag:
                cv2.imshow(self.tracker_type, frame)
            frame_count += 1
            # needs to wait unless the frame will not be displayed
            cv2.waitKey(1)
            # if the listener detects a keypress, we exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()    
        self.cap.release()

   

    
    # the default tracker is the comparison tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EmotionNet')

    parser.add_argument('--tracker', type=str, default='boring', help='tracker type')
    parser.add_argument('--camera', action=argparse.BooleanOptionalAction)
    parser.add_argument('--mask', action=argparse.BooleanOptionalAction)
    parser.add_argument('--emotion', action=argparse.BooleanOptionalAction)
    parser.add_argument('--resolution', type=int, default=256, help='resolution of the frame')

    args = parser.parse_args()
    tracker_type = args.tracker
    camera_flag = args.camera
    mask_flag = args.mask
    emotion_flag = args.emotion
    resolution = args.resolution

    emotionNet = EmotionNet(tracker_type)
    emotionNet.run(emotion_flag, mask_flag, camera_flag, resolution)

    
# to run use 
# Both mask and emotion
# python3 main.py --tracker boring --camera --mask --emotion


# # Only mask, no emotion
# python3 main.py --tracker boring --camera --mask --no-emotion

# # Only emotion in the top left corner
# python3 main.py --tracker boring --camera --no-mask --emotion

# # Only display, nothing in the terminal
# python3 main.py --tracker boring --camera --no-mask --no-emotion

# # Only emotions in the terminal, both do the same thing
# python3 main.py --tracker boring --no-camera --mask --emotion
# python3 main.py --tracker boring --no-camera --no-mask --emotion

# these do nothing and should not be allowed 
# python main.py --tracker boring --camera False --mask True --emotion False
# python main.py --tracker boring --camera False --mask False --emotion False   

