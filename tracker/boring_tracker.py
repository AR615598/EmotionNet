import cv2
import numpy as np

class boring_tracker:
    def __init__(self, cap) -> None:
        self.prev_frame = None
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    
        self.centroids = [self.width / 2, self.height / 2]
        self.avg_radius = 1
        
          
          
    
    def mask_shape(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray_frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # this is the difference between the current frame and the previous frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if faces is not None and len(faces) > 0:   
            a,b,c,d = faces[0]
            self.avg_radius = np.array([c,d])
            top_left = [a,b]
            bottom_right = [a+c, b+d]
            self.centroids = np.array([(top_left[0] + bottom_right[0])/2, (top_left[1] + bottom_right[1])/2])

        return self.centroids, self.avg_radius 
    




