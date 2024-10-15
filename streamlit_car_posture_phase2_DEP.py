import streamlit as st
import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np
import json
from PIL import Image
import io

def process_image(image_path):
    # Initialize progress bar
    progress = st.progress(0)

    # Load the model to detect truck parts
    progress.progress(25)
    model = YOLO(r'C:\Users\SwagatamRoy\Downloads\CAR POSTURE DETECT_oct_2024\trial3\best_discussed_with_TEAM_then_trained_14thoct_2024.pt')
    results = model.predict(image_path, conf=0.085, line_width=2)
    
    # Extract detection information
    progress.progress(50)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    labels = [model.names[int(class_id)] for class_id in class_ids]
    
    # Calculate areas of bounding boxes
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    
    # Find the index of the largest (nearest) object with the highest confidence
    if len(areas) > 0:
        nearest_highest_conf_index = max(range(len(areas)), key=lambda i: (areas[i], confidences[i]))
        
        # Prepare data for JSON output (only for the nearest object with highest confidence)
        label = labels[nearest_highest_conf_index]
        confidence = float(confidences[nearest_highest_conf_index])
        
        # Check if label is 'Passenger_Side' and confidence is less than 0.52
        if label == 'Passenger_Side' and confidence < 0.52:
            label = 'Front-right'
        
        output_data = [{
            'bounding_box': boxes[nearest_highest_conf_index].tolist(),
            'confidence': confidence,
            'label': label
        }]
        
        # Plot only the selected object
        image = cv2.imread(image_path)
        box = boxes[nearest_highest_conf_index].astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {confidence:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        plotted_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        output_data = []
        plotted_image = Image.open(image_path)
    
    # Convert the output data to JSON format
    output_json = json.dumps(output_data, indent=4)
    
    progress.progress(100)
    return plotted_image, output_json

# Streamlit app
st.title('Truck Posture Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    image = Image.open(uploaded_file)
    image_path = 'temp_image.png'
    image.save(image_path)
    
    if st.button('Process Image'):
        st.write("Processing...")
        plotted_image, result_json = process_image(image_path)
        
        # Display JSON result
        st.json(result_json)
        
        # Display the plotted image
        st.image(plotted_image, caption='Detected Truck Part', use_column_width=True)