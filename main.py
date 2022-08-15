import logging
import sys
import time

import cv2
from detector import MoveNetMultiPose
import utils
from config import *

def start():
    global cam
    global row_size
    global left_margin
    global text_color
    global font_size
    global font_thickness
    global fps_avg_frame_count
    global keypoint_detection_threshold_for_classifier
    
    pose_detector = MoveNetMultiPose(model, tracker)
    counter, fps = 0, 0
    start_time = time.time()
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight )
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read Any Camera ')

        counter += 1
        #image = cv2.flip(image, 1)
        list_persons = pose_detector.detect(image)
        image = utils.visualize(image, list_persons)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
          break
        cv2.imshow("Window", image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  start()
