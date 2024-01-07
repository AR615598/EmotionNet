# EmotionNet: Real-Time Emotion Classifier for Facial Images 📸🔍
An emotion classification project designed for real-time analysis using webcam input, containing a classification generator for standard use and an startup file for demonstration purposes.
## Overview 

This project leverages the user's webcam as a primary input, utilizing the [openCV](https://opencv.org/) library to process data on a frame-by-frame basis. Each frame undergoes tracking, where filters, masks, or neural networks are employed to estimate the center of the subject's face. Utilizing this information, a bounding box is approximated around the user, optimizing input size for the classifier while maintaining data integrity. Following the capture and storage of the specified photo subsection, the classification process is initiated. To acheive this, the [DeepFace](https://github.com/serengil/deepface) and [FER](https://github.com/JustinShenk/fer) classifier is employed for classifications. The user-configurable parameters dictate whether the emotion output is displayed on both the terminal and a real-time display or solely on the terminal.

## Dependencies 

- OpenCV 
- NumPy 
- Math 
- Sys 
- Logging 
- Typing
- FER
- DeepFace
- OS
- ArgumentParser
- Time 
- Main module 
- Inquirer 
- Regular Expressions 

## Example 
To run the example you will need to run the setup file as such 

```
python3 startup.py
```
Upon its first execution, this program will prompt you to authorize webcam access, ensuring the necessary permissions are granted. Following this, it will walk you through a series of configuration questions, allowing you to tailor the output to your preferences. If you wish to terminate the program while it's running, you can either disable the webcam feed or simply press the exit button. This action will conclude the execution of the run() function within the main program.
## Usage
This exmaple illustrates how we can use the project: 
``` python
import main.py 

tracker = 'cont'
camera_flag = True
mask_flag = True
emotion_flag = True
resolution = 100

emotions = []
emotionNet = main.EmotionNet(tracker)
for x,y in emotionNet.run(emotion_flag, mask_flag, camera_flag, resolution):
    emotions.append(x)
```
In this example, we utilized the cont tracker, and here are all available trackers:

- cont: This tracker compares contours in a set of recent frames, averaging them to identify the frame's subject. It is effective on a static background.
- comp: Similar to the contour tracker, this tracker averages absolute differences between current and previous frames to identify the subject. It is applicable under the same constraints as the cont tracker.
- NN: This tracker is a work in progress.
- boring: This tracker uses the native OpenCV facial detection model.

The `camera_flag` determines whether to display the live webcam feed. It can be complemented with the `mask_flag`, which, when set to true, will draw bounding boxes around the images received by the classifiers. The next parameter, `emotion_flag`, toggles the classifier on or off. Finally, the `resolution` value sets the dimensions of the input image. Lower resolutions are computationally less expensive and faster, but less accurate, with a minimum resolution limit of 20.

`emotionNet.run()` is a generator designed for aggregating observed emotions effortlessly. Grant the necessary permissions, specify your desired parameters, and initiate the process to begin aggregating emotion classifications seamlessly. 

To conclude the process, the same principles apply when quitting from the setup.py. You can either terminate the webcam feed or press the 'Esc' key. This will effectively exit the program and complete the execution of the setup process.


## Disclaimer 
This project operates entirely offline, ensuring no internet connection is required. All temporary files are automatically cleaned up upon the program's termination. In the event of an interruption, the temporary file can be accessed at 'classifiers/frame.jpg' and manually deleted if necessary.





