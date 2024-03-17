import os
import cv2
import pyttsx3 
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
t2s = pyttsx3.init()
import time
from tensorflow.python.keras.models import load_model
model = load_model('action1.h5', compile=False)

from flask import Flask, render_template, request

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


app=Flask(__name__ )

actions = np.array(['Hello','I am','Affan','Thanks', 'i love you','Fever','See you', 'God'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(100, 20, 70), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(100, 50, 20), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(200,100,100), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(200,50,50), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])




def detect_mediapipe():

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    global face_encodings
    global face_names

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                
                res = model.predict(np.expand_dims(sequence, axis=0))[-1]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                new_word = actions[np.argmax(res)]
                                # t2s.say(new_word)
                                # t2s.runAndWait()
                        else:
                            sentence.append(actions[np.argmax(res)])
                            new_word = actions[np.argmax(res)]
                            # t2s.say(new_word)
                            # t2s.runAndWait() 



                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV_Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/mediapipe_detect")
def mediapipe_detect():
    detect_mediapipe()

    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)
