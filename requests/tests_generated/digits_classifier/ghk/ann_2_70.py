import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For this example, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    images = []
    labels = []
    
    for i in range(num_images):
        img_path = f"tests/test_image_{i}.png"
        
        # Create a sample image and save it to disk if not already created.
        if not os.path.exists(img_path):
            img_data = np.random.randint(0, 256, size=(image_size[1], image_size[0]), dtype=np.uint8)
            Image.fromarray(img_data).save(img_path)
        
        # Load the saved image and append it to our test set.
        images.append(np.array(Image.open(img_path).convert('L').resize(image_size)))
        labels.append(i % 10)  # Assign a label (0-9) for each image
        
    return np.array(images), np.array(labels)

def test_digit_recognition_accuracy(model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    images, expected_labels = test_set
    
    classifier: IClassifyDigits = ClassifyDigits()
    predicted_labels = classifier(images)
    
    # Calculate the accuracy
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
