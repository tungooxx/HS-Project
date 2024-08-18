import streamlit as st
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keras.models import load_model
from tool import *

actions = np.array(["Xin chao","Cam on","Toi yeu ban","Thay","Xin loi"])

model = load_model('50frame.h5')

# Streamlit app title
st.title("Real-time Video")

# Start video capture
cap = cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Contain sequence
sequence = []
# Threshold
threshold = 0.7
sentence = []
prediction = []

# Layout
stframe = st.empty()
probability_display = st.empty()

# Streamlit video display
stframe = st.empty()
with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        image,result = mediapipe_detection(frame, holistic)
        
        keypoint = extract_keypoints(result)
        sequence.append(keypoint)
        sequence = sequence[-50:]
        if len(sequence) == 50:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction.append([np.argmax(res)])

            if np.unique(prediction[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold:                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
    
                if len(sentence) > 5:
                    sentence = sentence[-5:]
        # Display the frame with predictions
        stframe.image(frame, channels="BGR")
        
        # Display predictions
        probability_display.markdown("### Probabilities")
        for i, prob in enumerate(res):
            probability_display.markdown(f"{actions[i]}: {prob:.2f}")
        
        # Display sentence
        st.write(f"Sentence: {' '.join(sentence)}")
        
        
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()