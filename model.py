import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import cv2
from PIL import Image
import tensorflow as tf

Image_Width=32
Image_Height=32

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

def loadmodel(path):
    model = tf.keras.models.load_model(path)
    return  model


def preprocess_image(image):
    """"""""""
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((IMAGE_Width, IMAGE_Height))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert the image to a numpy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image 
    """""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.shape[2] == 4:
        # If the image has an alpha channel, remove it
        image = image[:, :, :3]

        # Ensure Image_Width and Image_Height are defined
    if not (Image_Width > 0 and Image_Height > 0):
        raise ValueError("Image_Width and Image_Height must be positive integers.")

        # Print for debugging
    print(f"Image shape before resize: {image.shape}")

    # Resize image
    image = cv2.resize(image, (Image_Width, Image_Height))

    print(f"Image shape after resize: {image.shape}")

    # Normalize and add batch dimension
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image