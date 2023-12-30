from typing import Any
from fer import FER
from deepface import DeepFace
import cv2

    # this class will classify the provided frame/image 
    # training will use a premade dataset of mixed emotions 
    # in actual use the frames from the tracker class. 
    # Generally it will collect every the bounding box from every 
    # 10 frames, the tracker calss will classify if the subset image 
    # contains a face/portrait. So all this class will do is classify 

    # sets up the model
class classifier:
    def __init__(self, frame=None) -> None:
        self.classifier = FER(mtcnn=True)
        self.default_image = '/Users/alexramirez/Personal_Projects/EmotionNet/data/photos/archive/test/happy/PrivateTest_258543.jpg'
        self.default_image_class = 'happy' 
        if frame is None: 
            pass


    def __call__(self, img = None) -> Any:
        if img == None:
            img = self.default_image
            img = cv2.imread(img)

        else:
            img = cv2.imread(img)
        return self.pred(img)


    def pred(self, img):
        dominant_emotion, emotion_score = self.classifier.top_emotion(img)
        # fer will return None if no face is detected
        # deepface will always return an emotion
        # but fer is more accurate
        final_pred = None
        # top emotion will return None if no face is detected
        if dominant_emotion != None:
            final_pred = dominant_emotion
        else:
            DeepF = DeepFace.analyze(img_path = img, enforce_detection=False, actions = ['emotion', 'dominant_emotion'])
            final_pred = DeepF[0]['dominant_emotion']
        return final_pred

if __name__ == "__main__":
    # test the classifier
    classifier_instance = classifier()
    classifier_instance()


    