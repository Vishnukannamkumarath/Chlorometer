import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import gdown
import os
import requests
from PIL import Image
import io
import altair as alt
import pandas as pd

# Google Drive model file setup
file_id = "1Z5rMuiycZcSYFy2Fh8lrt3fylnzEC5Qf"  # Replace with your file ID
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
file_id=["1x6bYU0pk2-VAcG3gONEWnu8cQ2SqRBLw","10dbB1rlFGZUNWsY8fMvG1KANYGcK9fZ1","1GJGRPVq7SukO7T2XxI59GOR8C7JAqbMG","1nfgqVHaCnDzXpDw4KpwUMuwCSrwfF_06","1WCq88fPLBFvsr5lFLIvKkUJ45vlxL7QU","1HMMn3mSR8BeogSdiYv_ymZjQOcfoT-bm"]
con=[0.5,0.7,1.0,1.5,2.0,2.5]
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

            # Display comments based on concentration range
            if predicted_concentration <= 0.2:
                st.error("Low level of Chlorine.Not Good for Drinking!")
            elif 0.2 < predicted_concentration <=0.5:
                st.info("It is Safe for use")
            else:
                st.error("High Level of Chlorine Not Good for Use!")
            st.subheader("Bar Chart")
            data=pd.DataFrame({
                 "Concentration":["Predicted"],
                  "Value":[predicted_concentration]
                })
            chart=alt.Chart(data).mark_bar(size=150).encode(
            x=alt.X("Concentration:N",title=""),
            y=alt.Y("Value:Q",title="concentration(mg/L)",scale=alt.Scale(domain=[0,4])),
            color=alt.value("steelblue")
            ).properties(
              width=300,
              height=400
            )
            st.altair_chart(chart,use_container_width=False)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
