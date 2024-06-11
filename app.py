pip install joblib

import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load your trained model
model = joblib.load('Covid19PREDICTION_CNN_model.pkl')

# Define the UI
st.title('COVID-19 Prediction App')
st.write('Fill in patient information and upload lung X-ray and masked images:')

# Example input fields
patient_name = st.text_input('Patient Name')
age = st.number_input('Age', min_value=0, max_value=150, value=30)

# Upload images
xray_image = st.file_uploader('Upload Lung X-ray Image', type=['jpg', 'jpeg', 'png'])
masked_image = st.file_uploader('Upload Masked Image', type=['jpg', 'jpeg', 'png'])

if xray_image and masked_image:
    # Process uploaded images
    xray_img = Image.open(xray_image)
    masked_img = Image.open(masked_image)

    # Make predictions
    # You'll need to preprocess the images (e.g., resize, normalize) before feeding them into the model
    prediction = model.predict([patient_name, age, xray_img, masked_img])

    # Display results
    st.write('Prediction:', prediction)
    st.image(xray_img, caption='Lung X-ray Image', use_column_width=True)
    st.image(masked_img, caption='Masked Image', use_column_width=True)





