import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Smart Fruit Classification",
    page_icon="🍎",
    layout="centered"
)

# ---------------------------------
# Settings
# ---------------------------------
IMG_SIZE = 100
CLASS_NAMES = ["Apple", "Banana", "Orange"]

# ---------------------------------
# Load CNN Model
# ---------------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        "model/cnn_model.keras"
    )

model = load_cnn_model()

# ---------------------------------
# Title
# ---------------------------------
st.title("🍎 Smart Fruit Classification System")

st.write(
    "Upload an image of an Apple, Banana, or Orange "
    "and the CNN model will predict its class."
)

st.divider()

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # ---------------------------------
    # Predict Button
    # ---------------------------------
    if st.button("🔍 Predict Fruit"):

        # Resize
        resized_image = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        # Convert to numpy array
        img_array = np.array(
            resized_image,
            dtype=np.float32
        )

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            np.max(predictions[0]) * 100
        )

        # ---------------------------------
        # Show Result
        # ---------------------------------
        st.success(
            f"Predicted Fruit: {predicted_class}"
        )

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        # Show probabilities
        st.subheader("Prediction Probabilities")

        for fruit, probability in zip(
            CLASS_NAMES,
            predictions[0]
        ):
            st.write(
                f"{fruit}: {probability * 100:.2f}%"
            )

st.divider()

st.caption(
    "Deep Learning CNN Image Classification Project"
)