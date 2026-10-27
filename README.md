# Smart Fruit Classification System Using CNN

## Project Overview
This project is a Deep Learning based image classification system
that identifies fruit images using a Convolutional Neural Network (CNN).

The system currently classifies images into three categories:
- Apple
- Banana
- Orange

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## CNN Architecture
The CNN model contains:
- Conv2D Layers
- MaxPooling2D Layers
- Flatten Layer
- Dense Layers
- Dropout Layer
- Softmax Output Layer

## Model Performance
Test Accuracy: 93.11%

The model was evaluated using:
- Accuracy
- Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Application
A Streamlit GUI allows users to upload a fruit image.

The trained CNN model predicts:
- Fruit Class
- Confidence Score
- Class Probabilities

## Classes
1. Apple
2. Banana
3. Orange

## Run Application

```bash
streamlit run app.py