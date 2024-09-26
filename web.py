from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import io
from PIL import Image, ImageDraw, ImageFont
import sys
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from IPython.display import Image
import urllib.parse
import requests
import os
# Try to import TensorFlow and imageio, with fallback options
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("TensorFlow not found. Please install it using: pip install tensorflow")
    print("Using a placeholder for the model.")
    def load_model(path):
        print(f"Placeholder: Model loaded from {path}")
        return None

from tool import *

app = Flask(__name__)
actions = np.array(["Ky Nang","Cam on","May Man","Ruc Ro","Thay","Tu Choi","Xin chao","Xin loi"])
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
# Threshold
threshold = 0.85
sentence = []
prediction = []
sequence = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'view_webcam' in request.form:
            return redirect(url_for('webcam'))
        elif 'view_text_to_gif' in request.form:
            return redirect(url_for('text_to_gif'))
    return render_template('index.html')

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    if request.method == 'POST':
        if 'back_to_main' in request.form:
            return redirect(url_for('index'))
    return render_template('webcam.html')

@app.route('/text_to_gif', methods=['GET', 'POST'])
def text_to_gif():
    if request.method == 'POST':
        if 'back_to_main' in request.form:
            return redirect(url_for('index'))
        
        elif 'translate_gif' in request.form:
            text = request.form['text']
            print(text)
            # Translate spoken text to signed pose
            url = translate_spoken_to_signed(text)
            response = requests.get(url)

            # Save the pose data to a .pose file
            pose_filename = 'output.pose'
            with open(pose_filename, 'wb') as file:
                file.write(response.content)

            # Read pose data and create a visualizer
            data_buffer = open(pose_filename, "rb").read()
            pose = Pose.read(data_buffer)
            visualizer = PoseVisualizer(pose)

            # Save the visualized output as a GIF
            gif_filename = "static/gif/output.gif"
            os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
            visualizer.save_gif(gif_filename, visualizer.draw())
            
            return render_template('text_to_gif.html', gif_filename='gif/output.gif')
        else:
            print("GG!")

    return render_template('text_to_gif.html')


def translate_spoken_to_signed(text: str, spoken_language: str = 'en', signed_language: str = 'ase') -> str:
    api = 'https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose'
    query = f"?text={urllib.parse.quote(text)}&spoken={urllib.parse.quote(spoken_language)}&signed={urllib.parse.quote(signed_language)}"
    return f"{api}{query}"

def generate_frames():
    global sequence, sentence, prediction
    camera = cv2.VideoCapture(0)  # 0 is the default camera
    if not camera.isOpened():
        raise RuntimeError("Could not start the camera.")
    try:
        model = load_model('30new.h5')
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
    
    with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        while True:
            success, frame = camera.read()  # Read the frame from the camera
            if not success:
                break

            image, result = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, result)
            keypoints = extract_keypoints(result)
            
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30 and model is not None:
                try:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction.append(np.argmax(res))
                    
                    if np.unique(prediction[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                except Exception as e:
                    print(f"Error predicting: {str(e)}")
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_action')
def get_action():
    return jsonify({"action": sentence})

def generate_gif_from_text(text):
    if not imageio:
        raise ImportError("imageio is required for GIF generation")
    
    frames = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except IOError:
        print("DejaVuSans-Bold font not found. Using default font.")
        font = ImageFont.load_default()
    
    for char in text:
        image = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(image)
        w, h = draw.textsize(char, font=font)
        draw.text(((200-w)/2, (200-h)/2), char, font=font, fill='black')
        frames.append(np.array(image))
    
    output = io.BytesIO()
    imageio.mimsave(output, frames, format='GIF', duration=0.5)
    output.seek(0)
    return output

if __name__ == "__main__":
    app.run(debug=True)