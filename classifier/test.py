from fer import FER
import cv2
import numpy
emotion_detector = FER(mtcnn=True)
img = cv2.imread('/Users/alexramirez/Personal_Projects/EmotionNet/data/photos/archive/train/disgust/Training_659019.jpg')
cv2.imshow('img', img)
cv2.waitKey(0)
print(emotion_detector.detect_emotions(img))
