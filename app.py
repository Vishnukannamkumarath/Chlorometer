import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
model_path = '/home/vishnu-k/project/your_model.keras'  # Update with the actual path to your saved .keras model
model = tf.keras.models.load_model(model_path)

# Load the scaler
with open('/home/vishnu-k/project/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
    img = load_img(uploaded_file, target_size=(128, 128))  # Resize to model input size
    img_array = img_to_array(img)  # Convert to array

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Run prediction
    if st.button("Predict Concentration"):
        predicted_concentration = predict_image(model, img_array)
        st.success(f'Predicted concentration: {predicted_concentration:.4f}')
