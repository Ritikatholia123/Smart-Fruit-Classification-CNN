import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Settings
IMG_SIZE = 100
BATCH_SIZE = 32
TEST_PATH = "dataset/test"

# Load model
model = tf.keras.models.load_model("model/cnn_model.keras")

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_data.class_names

print("\nClasses:", class_names)

# Normalize images
test_data_normalized = test_data.map(
    lambda x, y: (x / 255.0, y)
)

# Actual labels
y_true = []

for images, labels in test_data:
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)

# Predictions
predictions = model.predict(test_data_normalized)

y_pred = np.argmax(predictions, axis=1)

# Classification Report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title("CNN Confusion Matrix")
plt.savefig(
    "model/confusion_matrix.png",
    bbox_inches="tight"
)

plt.show()