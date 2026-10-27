import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

# Model folder create
os.makedirs("model", exist_ok=True)


# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)


# ==========================================
# LOAD TEST DATA
# ==========================================

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_data.class_names

print("\nClasses:", class_names)
print("Number of Classes:", len(class_names))


# ==========================================
# IMAGE NORMALIZATION
# ==========================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

test_data = test_data.map(
    lambda x, y: (normalization_layer(x), y)
)


# ==========================================
# CNN MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\nCNN MODEL SUMMARY")
model.summary()


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)


# ==========================================
# EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

model.save("model/cnn_model.keras")

print("\nModel saved successfully!")
print("Location: model/cnn_model.keras")


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("model/accuracy_graph.png")
plt.show()


# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("model/loss_graph.png")
plt.show()