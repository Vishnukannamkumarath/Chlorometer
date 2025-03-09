import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import gdown
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io
import altair as alt
import pandas as pd

# Google Drive model file setup
file_id = "1wQI8ojNfFX5lUU6zT3cLIw9XNPETbQ2O"  # Replace with your file ID
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
scaler_path = "c64scaler.pkl"  # Ensure the scaler is in your directory
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
st.subheader("Instructions for use")
instructions=[
    "Take 100ml of Water Sample for testing.","Add 5ml of DPD reagent into it.","Mix the reagent thoroughly to ensure complete reaction.","Take a photo of the resultant sample and upload it."
]
st.write(instructions)
#Image for attraction
#load images
@st.cache_data
def load_image(image_url):
    response=requests.get(image_url)
    return Image.open(io.BytesIO(response.content))

st.subheader("Chlorine Concentration Levels(mg/L)")
file_id=["17pC37F2tp1xi1aY7k8fEldypX2QU7eqI","1_sUrW976QDXWAAnI8Zy2ICcva7zfyiVZ","1BO7ZBhsjfp_XJsEpno4S3qLiSH8Yccon","17biu6bPrROo2YMfbqmo8OaBZRAW2pdQM","1HdNzXV5VHw8UIe0P3CattlquLwMnH_Nk","1Kj1wtnTSPp3nh6GsBCs7-b_jfIhgHIRc","1WNrJLuEwi34A6WwEn9As3UAqw8DV5HOo"]
con=[0.1,0.2,0.5,1.0,1.5,2.0,2.5]
cols=st.columns(len(file_id))
for idx,file_id in enumerate(file_id):
    image_url = f"https://drive.google.com/uc?export=view&id={file_id}"
    con_value=con[idx]
    with(cols[idx]):
        img=load_image(image_url)
        st.image(img,caption=f"At:{con_value}mg/L",use_column_width=True)
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

            
            if predicted_concentration <= 0.2:
                st.error("Low level of Chlorine.Not Good for Drinking!")
            elif 0.2 < predicted_concentration <=0.5:
                st.info("It is Safe for use")
            else:
                st.error("High Level of Chlorine Not Good for Use!")
            
            st.title("pH Prediction")
            st.subheader("pH vs. Concentration Chart")
            concentration_values = np.round(np.linspace(0.1, 2.5, 25), 2).tolist()
            ph_values = [5.85,5.55,5.37,5.25,5.15,5.07,5.00,4.95,4.90,4.85,4.81,4.77,4.74,4.70,4.67,4.65,4.62,4.59,4.57,4.55,4.53,4.51,4.49,4.47,4.45]  
            closest_concentration = min(concentration_values, key=lambda x: abs(x - predicted_concentration))
            predicted_ph = ph_values[concentration_values.index(closest_concentration)]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(concentration_values, ph_values, marker="o", linestyle="-", color="gray", alpha=0.5, label="Full Data")
            ax.scatter([closest_concentration], [predicted_ph], color="red", s=100, label="Predicted Point")
            ax.set_xlabel("Concentration", fontsize=12)
            ax.set_ylabel("Predicted pH", fontsize=12)
            ax.set_title("Predicted Concentration vs pH", fontsize=14)
            ax.set_xticks(concentration_values[::2])  
            ax.tick_params(axis="x", rotation=45)
            ax.set_yticks(ph_values)  
            ax.legend()
            st.pyplot(fig)
            st.write(f"### 🔹 Predicted Concentration : **{predicted_concentration:.3f}**")
            st.write(f"### 🔹 Mapped to Nearest: **{closest_concentration}**")
            st.write(f"### 🔹 Corresponding pH Value: **{predicted_ph}**")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
