import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from tensorflow.keras.models import load_model
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_DATA_PATH
import matplotlib.pyplot as plt
import pickle
import os
import pytest

# Load test data and labels from file (assuming it's stored in a pickle)
def load_test_data():
    with open(TEST_DATA_PATH + 'test_images.pkl', 'rb') as f:
        images = pickle.load(f)

    with open(TEST_DATA_PATH + 'test_labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    return np.array(images), np.array(labels)


# Load model
model = load_model(MODEL_DIGIT_RECOGNITION_PATH)


def test_classification_accuracy():
    # Get the test data and labels
    images, labels = load_test_data()

    # Normalize and flatten the input data
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Make predictions on the test set using our model
    predictions = np.argmax(model.predict(flattened_images), axis=1)
    
    # Calculate accuracy by comparing predicted labels with actual ones
    correct_predictions = sum(predictions == labels)
    total_samples = len(labels)
    accuracy = (correct_predictions / total_samples) * 100

    assert accuracy >= 95, f"Model's classification accuracy is {accuracy:.2f}%, which is less than the required threshold of 95%"


def test_model_output_shape():
    # Get a single image from the test data
    images, _ = load_test_data()
    
    # Normalize and flatten the input data
    normalized_images = images / 255.0
    flattened_image = normalized_images[0].reshape(1, -1)

    prediction = model.predict(flattened_image)
    assert len(prediction.shape) == 2, "Model output should have two dimensions"
    assert prediction.shape[-1] == 10, f"Expected the last dimension of model's output to be 10 (for digits from 0-9), but got {prediction.shape[-1]}"
