import streamlit as st
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from PIL import Image
import tempfile

# Define text style and color
text_color = "#4a4a4a"  # Dark gray for professional look
text_style = "font-family: Arial, sans-serif; font-size: 18px; font-weight: bold;"

# Initialize Firebase app
def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(r"C:\Users\S SRIVARSHINI\Downloads\igniters-street-auto-firebase-adminsdk-uuiih-533b0bb74d.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://your-database-url.firebaseio.com/'  # Update this URL
        })
    return firestore.client()

# Streamlit app
st.set_page_config(page_title="YOLO Object Detection", layout="wide")

# Header and subheader
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔍 Person and Vehicle Detection with YOLO</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #ff7f0e;'>📊 Detecting and Sending Data to Firebase</h2>", unsafe_allow_html=True)

# Upload a file
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Define class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']

    if file_extension in ['jpg', 'jpeg', 'png']:
        # Load image
        image = Image.open(uploaded_file)
        image_path = 'uploaded_image.jpg'
        image.save(image_path)

        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Load a pretrained model

        # Perform inference on the uploaded image
        results = model(image_path, save=True)

        # Initialize counters for detected classes
        counts = {name: 0 for name in class_names}
        total_vehicles = 0

        # Iterate through each detection result
        for result in results:
            boxes = result.boxes
            cls = boxes.cls.tolist()

            # Count occurrences of each class
            for name in class_names:
                counts[name] += cls.count(class_names.index(name))

        # Adjust person count to exclude bicycle and motorcycle detections
        counts['person'] -= (counts['bicycle'] + counts['motorcycle'])
        total_vehicles = counts['bicycle'] + counts['car'] + counts['bus'] + counts['motorcycle']

        # Display results with consistent style
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of people detected: {counts['person']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of bicycles detected: {counts['bicycle']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of cars detected: {counts['car']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of motorcycles detected: {counts['motorcycle']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of buses detected: {counts['bus']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Total vehicles detected: {total_vehicles}</h3>", unsafe_allow_html=True)

        # Push data to Firebase
        db = initialize_firebase()
        ref = db.collection('AI').document('chromepet')

        ref.set(counts)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Data pushed successfully!</h3>", unsafe_allow_html=True)

    elif file_extension in ['mp4', 'avi', 'mov']:
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load YOLO model
        model = YOLO("yolov8n.pt")  # Load a pretrained model

        # Process video
        cap = cv2.VideoCapture(temp_file_path)
        stframe = st.empty()  # Placeholder for displaying video frames

        # Initialize counters
        counts = {name: 0 for name in class_names}
        total_vehicles = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on each frame
            results = model(frame)

            # Reset counters for each frame
            frame_counts = {name: 0 for name in class_names}

            # Iterate through each detection result
            for result in results:
                boxes = result.boxes
                cls = boxes.cls.tolist()

                # Count occurrences of each class
                for name in class_names:
                    frame_counts[name] += cls.count(class_names.index(name))

            # Adjust person count to exclude bicycle and motorcycle detections
            frame_counts['person'] -= (frame_counts['bicycle'] + frame_counts['motorcycle'])
            total_vehicles = frame_counts['bicycle'] + frame_counts['car'] + frame_counts['bus'] + frame_counts['motorcycle']

            # Display results with consistent style
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of people detected: {frame_counts['person']}</h3>", unsafe_allow_html=True)
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of bicycles detected: {frame_counts['bicycle']}</h3>", unsafe_allow_html=True)
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of cars detected: {frame_counts['car']}</h3>", unsafe_allow_html=True)
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of motorcycles detected: {frame_counts['motorcycle']}</h3>", unsafe_allow_html=True)
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Number of buses detected: {frame_counts['bus']}</h3>", unsafe_allow_html=True)
            stframe.markdown(f"<h3 style='color: {text_color}; {text_style}'>Total vehicles detected: {total_vehicles}</h3>", unsafe_allow_html=True)

        cap.release()

        # Push final data to Firebase
        db = initialize_firebase()
        ref = db.collection('AI').document('chromepet')

        ref.set(counts)
        st.markdown(f"<h3 style='color: {text_color}; {text_style}'>Data pushed successfully!</h3>", unsafe_allow_html=True)
