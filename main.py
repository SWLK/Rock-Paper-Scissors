###############
###load data###
###############

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import mediapipe as mp
import cv2
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

# X_train = np.load('train_data.npy')
# y_train = np.load("train_label.npy")
# print(y_train[100])
# plt.imshow(X_train[100])
# plt.show()

# evaluate whether finger is extended
def extended(finger):
    # data type [[],[],[],[],[]]
    current_y = 1
    for point in finger:
        if point[1]<current_y:
            current_y = point[1]
        else:
            return False
    return True

def eval_hand(hand):
    # data type [[],[],[],[],[]]
    thumb = hand[1:5]
    ind = hand[5:9]
    mid = hand[9:13]
    ring = hand[13:17]
    pink = hand[17:21]

    arr = [extended(thumb),extended(ind),extended(mid),extended(ring),extended(pink)]
    #print(arr)
    # rock
    if arr == [True or False,False,False,False,False]:
        segment = AudioSegment.from_wav("voices/paper.wav")
        play(segment)
        return 0

    # paper
    elif arr == [True,True,True,True,True]:
        segment = AudioSegment.from_wav("voices/scissors.wav")
        play(segment)
        return 1

    # scissors
    elif arr == [True or False,True,True,False,False]:
        segment = AudioSegment.from_wav("voices/rock.wav")
        play(segment)
        return 2

    # other
    else:
        return 4


# video variables  ------------------------
FRAME_WIDTH = 600
FRAME_HEIGHT = 600

# detection object ------------------------
hand_detector = mp.solutions.hands
hand = hand_detector.Hands(max_num_hands=1,min_detection_confidence=0.7)
draw_hand = mp.solutions.drawing_utils

# webcam capture --------------------------
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    ret,frame = cam.read()
    # converting image to diff color scheme and pass to hand detector
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hand.process(rgb_frame)

    if res.multi_hand_landmarks:

        for hand_landmark in res.multi_hand_landmarks:
            hand_points = []

            # iterate through each point in hand
            for point in hand_detector.HandLandmark:
                normalizedLandmark = hand_landmark.landmark[point]
                pixelCoordinatesLandmark = draw_hand._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, FRAME_WIDTH, FRAME_HEIGHT)

                # storing the points of a hand
                if pixelCoordinatesLandmark:
                    x = pixelCoordinatesLandmark[0]/FRAME_WIDTH
                    y = pixelCoordinatesLandmark[1]/FRAME_HEIGHT
                    tup = (x,y)
                    hand_points.append(tup)

            hand_points = np.array(hand_points)
            #print(hand_points)
            #print(extended(hand_points[5:8]))
            print(eval_hand(hand_points))
            draw_hand.draw_landmarks(frame,hand_landmark,connections=hand_detector.HAND_CONNECTIONS)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


