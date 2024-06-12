import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the Keras model
model = load_model('Covid_CNN_Model.keras')

# Function to preprocess input image
def preprocess_input_image(image_data):
    # Resize image to match model input shape
    img = image.load_img(image_data, target_size=(224, 224))
    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img_array = img_array / 255.  # Normalize pixel values
    return processed_img_array

# Function to make predictions
def make_prediction(input_image):
    preprocessed_image = preprocess_input_image(input_image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit app
def main():
    st.title('Image Classification with Keras Model')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make prediction if image is uploaded
    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Predict'):
            prediction = make_prediction(uploaded_file)
            st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()






