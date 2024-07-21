#  Gets the computer permissions and asks for the tracker presets 
import sys
from main import EmotionNet
import inquirer
import cv2


ascii_title = """
███████╗███╗░░░███╗░█████╗░███╗░░██╗███████╗████████╗
██╔════╝████╗░████║██╔══██╗████╗░██║██╔════╝╚══██╔══╝
█████╗░░██╔████╔██║██║░░██║██╔██╗██║█████╗░░░░░██║░░░
██╔══╝░░██║╚██╔╝██║██║░░██║██║╚████║██╔══╝░░░░░██║░░░
███████╗██║░╚═╝░██║╚█████╔╝██║░╚███║███████╗░░░██║░░░
╚══════╝╚═╝░░░░░╚═╝░╚════╝░╚═╝░░╚══╝╚══════╝░░░╚═╝░░░
"""
def cam_permission():
    print("please grant access to the camera.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera. Please check if the camera is connected.")
        return False
    cap.release()
    return True


if __name__ == "__main__":
    # This will be controlling the interactive setup

    # Print out the title 
    print(ascii_title)
    # Ask permissions for the camera 
    perm = cam_permission()
    if perm == False:
        sys.exit()
    else:
        print("Camera permissions granted!")
    # Which tracker to use
    questions = [
        inquirer.List('size',
                    message="What tracker do you need?",
                    choices=['Frame Comparison Tracker', 'Contour Tracker', 'Neural Network (WIP)', 'CV2 Built in tracker (BEST)'],
                ),
        # then we will ask for the display preferences
        inquirer.List('display',
                    message="Display camera feed?",
                    choices=['Yes', 'No']
                ),
        # then we will ask for the mask preferences
        inquirer.List('mask',
                    message="Display bounding box on feed?",
                    choices=['Yes', 'No']
                ),
        # then we will ask for the emotion preferences
        inquirer.List('classifier',
                    message="Use emotion classifier?",
                    choices=['Yes', 'No']
                ),
        # then we will ask for the resolution preferences
        inquirer.Text('resolution', message="What resolution should the classifier use? (No less than 20)"),
    ]
    ans = inquirer.prompt(questions)
    trackers = {'Frame Comparison Tracker' : 'comp'
                , 'Contour Tracker' : 'cont'
                , 'Neural Network (WIP)' : 'NN'
                , 'CV2 Built in tracker (BEST)' : 'boring'}

    bools = {'Yes' : True, 'No' : False}
    tracker = trackers[ans['size']]
    camera_flag = bools[ans['display']]
    mask_flag = bools[ans['mask']]
    emotion_flag = bools[ans['classifier']]
    resolution = int(ans['resolution'])
    # run the main program

    emotionNet = EmotionNet(tracker)
    # generator dummy
    for x in emotionNet.run(emotion_flag, mask_flag, camera_flag, resolution):
        pass


