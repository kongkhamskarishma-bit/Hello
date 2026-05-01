import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ==== Configuration ====
P, Q = 128,128  # Example size: change if needed
target = {
    0:'awaphadigom', 1: 'mayangton', 2: 'peeruk', 3:'phakpai', 4:'tuningkhok'
    # Add more classes if needed
}

# ==== Load model ====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# ==== Title ====
st.title("Manipuri Herb Classification")

# ==== Upload Image ====
uploaded_file = st.file_uploader("Upload a color image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=200)

    # Resize and preprocess image
    img_resized = image.resize((Q, P))  # Width x Height
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = img_array.reshape(1, P, Q, 3)  # add batch dim

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = target.get(predicted_index, "Unknown")
    confidence = prediction[predicted_index]

    # Display results
    st.subheader("Prediction")
    st.write(f"**Predicted Label:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    st.subheader("Confidence for All Classes")
    for i, prob in enumerate(prediction):
        label = target.get(i, f"Class {i}")
        st.write(f"{label}: {prob:.2%}")
