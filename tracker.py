import cv2
import numpy as np
import time
import math
import sys



# Create a VideoCapture object and creates a mask for the subject
# in the frame. The mask is used to track the subject in the frame.
# init - creates a VideoCapture object
# create_mask - creates a mask for the subject in the frame
# display_camera - displays the mask in a window named "Camera"
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
            print(f"centroids: {centroids} avg_radius: {avg_radius}")
            cv2.circle(frame, (centroids[1], centroids[0]), 20, (0, 0, 255), cv2.FILLED)

            # Draw a circle with the specified center and radius
            cv2.circle(frame, (centroids[1], centroids[0]), avg_radius, (0, 0, 255), 2)



            # Display the frame in a window named "Camera"
            cv2.imshow("Camera", frame)
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
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
        frame_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1] 


        centroids = self.get_center(frame_diff)
        avg_radius = self.get_avg_radius(frame_diff, centroids)


        self.prev_frame = gray_frame
        return centroids, avg_radius





    # get_avg_radius numpy loop version
    def get_avg_radius_loop(self, frame, center):
        avg_radius = 0
        for x in range(len(frame)):
            for y in range(len(frame[0])):
                if frame[x][y] > 0:
                    avg_radius += math.sqrt((x - center[0])**2 + (y - center[1])**2)
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
        avg_x, avg_y = np.nonzero(frame) # get the average x value of all the nonzero values in the frame
        avg_y = avg_y.mean() # get the average y value of all the nonzero values in the frame
        avg_x = avg_x.mean() # get the average x value of all the nonzero values in the frame
        avg_radius = np.sqrt((avg_x - center[0])**2 + (avg_y - center[1])**2) # calculate the average radius
        return avg_radius
        
    # get_avg_radius numpy verctorizer version
    
    def get_center(self, frame):
        return np.nonzero(frame)[0].mean(), np.nonzero(frame)[1].mean() # get the average x and y values of all the nonzero values in the frame






if __name__ == "__main__":
    tracker = Tracker()
    tracker.display_camera()
