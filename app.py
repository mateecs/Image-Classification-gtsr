import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
from model import loadmodel, classes, preprocess_image
import matplotlib.pyplot as plt

model = loadmodel('latest model.h5')
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction, axis=1)[0]  # Get the class ID
    return prediction[0], class_id

def count_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    count = 0
    for file_name in os.listdir(directory):
        if os.path.splitext(file_name)[1].lower() in image_extensions:
            count += 1
    return count

def Labels(data_dir):
    labels = []
    data_dir = os.path.join(data_dir, "Train")
    for class_id in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_id)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                    labels.append(class_id)
    return np.array(labels)



with st.sidebar:

    st.title("Choose Action")
    choice = st.radio("Navigation", ["Predict","Dataset Analysis", "Model Score"])
    st.info("This project application helps you build and explore your data.")

if choice == "Predict":
    st.title("Image Classification App")
    st.write("Upload an image and get the class ID predicted by the model.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        prediction, class_id = predict(image)
        st.write(f"Predicted Class ID: {class_id}")
        st.write(f"Predicted sign: {classes[class_id]}")

        # Plotting the confidence plot
        st.write("Confidence Plot:")
        fig, ax = plt.subplots()
        ax.bar(range(len(classes)), prediction, color='blue')
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Scores for Each Class')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([i for i in range(len(classes))], rotation=45, fontsize=6)
        st.pyplot(fig)

