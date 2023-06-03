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
            self.centroids, self.avg_radius = self.get_bounds(frame, sorted_contours, n)

            # self.draw_n_contours(n, sorted_contours, frame)

        # Show the image
        # cv2.imshow("Threshold Image", threshold_image)
        self.prev_frame = gray_frame

        return self.centroids, self.avg_radius
    

    def draw_n_contours(self, n, contours, frame):
        for i in range(n):
            cv2.drawContours(frame, contours, i, (0,255,0), 3)
        



    # get_avg_radius numpy verctorizer version

    def get_bounds(self, frame, contours, n):
        def get_extremes(contour, center):
            contour = np.squeeze(contour)
            max_x = np.max(contour[:,0])
            max_y = np.max(contour[:,1])
            max_x_dist = np.abs(max_x - center[1])
            max_y_dist = np.abs(max_y - center[0])
            return [max_x_dist, max_y_dist]
        
        def get_avg_radius(self, center, contour):
            # list is in shape [[[x,y]],[[x,y]],[[x,y]]]
            contours_array = np.squeeze(contour)
            avg_offset = np.mean(np.sqrt((contours_array - center)**2), axis=0)
            return avg_offset.tolist()
        
        def get_contour_center(contour):
            # needs a way to handle noise 
            # we know a contour is noise if it is a single point
            # maybe even if it is two points
            squeezed = np.squeeze(contour)
            if len(squeezed) <= 100:
                return [np.nan, np.nan]
            
            center = np.mean(squeezed, axis=0)
            return center
        

        split_contours = (contours[:n])
        centers = []
        radiuses = []
        if len(split_contours) <= 1:
            center = get_contour_center(split_contours[0])
            centers = [center]
            rad = get_avg_radius(self, center, split_contours[0])
            radiuses = [rad]
        else:
            # i have no idea why but np.vectorize(get_contour_center) doesn't work
            # it begins to iterateover the indivdual coordinates of the points rather than 
            # the points themselves
            for contour in split_contours:
                center = get_contour_center(contour)
                centers.append(center)
        # check if there are any nan values
        # if there are, then we need to use the previous center
        if np.isnan(centers).any():
            avg_center = self.centroids      
            avg_radius = self.avg_radius

        else:
            avg_center = np.mean(centers, axis=0)
            avg_center = avg_center.astype(int)
            for contour in split_contours:
                # rad = get_avg_radius(self, avg_center, contour)
                rad = get_extremes(contour, avg_center)
                radiuses.append(rad)
            avg_radius = np.mean(radiuses, axis=0)

            avg_radius = avg_radius.astype(int)
            # print("avg_center: ", avg_center)
        return avg_center, avg_radius
    
