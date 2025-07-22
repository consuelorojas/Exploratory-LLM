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
    """Generate a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Create an array to hold the images and their corresponding labels.
    images = np.zeros((num_images,) + image_size)
    labels = np.random.randint(0, 10, size=num_images)

    for i in range(num_images):
        img_path = f"tests/test_image_{i}.png"
        
        # Create a simple test image with the label as its pixel value.
        img_array = np.full(image_size, labels[i], dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

    yield images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model):
    """Test that the digit recognition model achieves an accuracy of more than 95%."""
    
    # Generate a sample test set.
    _, _ = np.zeros((10,) + (28 * 28)), np.random.randint(0, 10, size=10)
    
    images = np.array([np.full((1, 784), i) for i in range(10)])
    labels = [i % 10 for i in range(10)]
    
    # Create an instance of the ClassifyDigits class.
    classifier: IClassifyDigits = ClassifyDigits()
    
    predictions = classifier(images)
    
    accuracy = np.mean(predictions == labels)

    assert accuracy > 0.95, f"Expected accuracy to be more than 95%, but got {accuracy:.2f}"
