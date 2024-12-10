import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import gdown
import os

# Google Drive model file setup
file_id = "1fZA70MwInBeWKpYqG0nVa7IISYfbxpzA"  # Replace with your file ID
download_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "model.keras"

# Download the model if not already present locally
if not os.path.exists(model_path):
    st.write("Downloading model file...")
    try:
        gdown.download(download_url, model_path, quiet=False)
        st.write("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# Load the scaler
scaler_path = "scaler.pkl"  # Ensure the scaler is in your directory
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    st.write("Scaler loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the scaler: {e}")
    st.stop()

# Function to predict image concentration
def predict_image(model, img_array):
    # Normalize and expand dimensions for the model
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]  # Rescale back to original value

# Streamlit app layout
st.title("ChloroMeter")
# Upload image functionality
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    try:
        img = load_img(uploaded_file, target_size=(128, 128))  # Resize to model input size
        img_array = img_to_array(img)  # Convert to array

        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Run prediction
        if st.button("Predict Concentration"):
            predicted_concentration = predict_image(model, img_array)
            st.success(f'Predicted concentration: {predicted_concentration:.4f}')
    except Exception as e:
        st.error(f"Error processing the image: {e}")
