import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 100

CLASS_NAMES = ["Apple", "Banana", "Orange"]

# -----------------------------
# Load trained CNN model
# -----------------------------
model = tf.keras.models.load_model(
    "model/cnn_model.keras"
)

print("Model loaded successfully!")


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path):

    if not os.path.exists(image_path):
        print("Image not found!")
        return

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert image to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = np.max(predictions[0]) * 100

    # Results
    print("\n==============================")
    print("FRUIT PREDICTION RESULT")
    print("==============================")

    print("Predicted Fruit :", predicted_class)
    print(f"Confidence      : {confidence:.2f}%")


# -----------------------------
# User input
# -----------------------------
image_path = input(
    "\nEnter image path: "
).strip().strip('"')

predict_image(image_path)