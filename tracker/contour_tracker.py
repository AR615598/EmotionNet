import cv2
import numpy as np

class contour_tracker:
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
        # this is the difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
        threshold_image = cv2.threshold(frame_diff, 35, 255, cv2.THRESH_BINARY)[1]


        # Find the contours in the image
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_NONE)
        length = len(contours)
        

        # Draw the contours on the image
        if length > 0:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if length > 5:
                n =5 
            else:
                n = length
            self.centroids = self.get_center(frame, sorted_contours, n)
            self.draw_n_contours(n, sorted_contours, frame)
            # avg_radius, avg_offset = self.get_avg_radius(frame, center, sorted_contours)

        # Show the image
        cv2.imshow("Threshold Image", threshold_image)
        self.prev_frame = gray_frame

        return self.centroids, 100
    

    def draw_n_contours(self, n, contours, frame):
        for i in range(n):
            cv2.drawContours(frame, contours, i, (0,255,0), 3)
        


    # np.vectorize(func) will return a function that can be applied to a numpy array
    def get_avg_radius(self, frame, center, contours):
        # list is in shape [[[x,y]],[[x,y]],[[x,y]]]
        contours_array = np.squeeze(contours)
        distances = np.sqrt(np.sum((contours_array - center) ** 2, axis=1))
        avg_radius = np.mean(distances)
        avg_offset = np.mean(contours_array - center, axis=0)
        return avg_radius, avg_offset.tolist()

        
        

    # get_avg_radius numpy verctorizer version

    def get_center(self, frame, contours, n):
        def get_contour_center(contour):
            squeezed = np.squeeze(contour)
            if len(squeezed.shape) == 1:
                return squeezed
                
            center = np.mean(squeezed, axis=0)
            return center

        split_contours = (contours[:n])
        centers = []
        if len(split_contours) <= 1:
            center = get_contour_center(split_contours[0])
            centers = [center]
        else:
            # i have no idea why but np.vectorize(get_contour_center) doesn't work
            # it begins to iterateover the indivdual coordinates of the points rather than 
            # the points themselves
            for contour in split_contours:
                center = get_contour_center(contour)
                centers.append(center)
        avg_center = np.mean(centers, axis=0)
        int_center = avg_center.astype(int)
        return int_center    
    

        # avg_x = np.sum(contours, axis=0) // len(contours)
        # avg_y = np.sum(contours, axis=1) // len(contours)
        # return [avg_x, avg_y]
        pass
    def draw_bounding_box():
        pass


