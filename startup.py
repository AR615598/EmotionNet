#  Gets the computer permissions and asks for the tracker presets 
import os
import sys
import time
import main
import inquirer
import re
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
                message="What size do you need?",
                choices=['Frame Comparison Tracker', 'Contour Tracker', 'Neural Network (WIP)', 'CV2 Built in tracker'],
            ),
    # then we will ask for the display preferences
    inquirer.List('size',
                message="Display camera feed?",
                choices=['Yes', 'No']
            ),
    # then we will ask for the mask preferences
    inquirer.List('size',
                message="Display mask?",
                choices=['Yes', 'No']
            ),
    # then we will ask for the emotion preferences
    inquirer.List('size',
                message="Use emotion classifier?",
                choices=['Yes', 'No']
            ),
    # then we will ask for the resolution preferences
    inquirer.Text('resolution', message="What resolution would you like to use?"),
]
inquirer.prompt(questions)

# then we will ask for the mask preferences
# then we will ask for the emotion preferences
# then we will ask for the resolution preferences
# then we will run the classifier 
