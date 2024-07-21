from emotionnet import main

tracker = 'cont'
camera_flag = True
mask_flag = True
emotion_flag = True
resolution = 100

emotions = []
emotionNet = main.EmotionNet(tracker)
for emotion,confidence in emotionNet.run(emotion_flag, mask_flag, camera_flag, resolution):
    emotions.append(emotion)