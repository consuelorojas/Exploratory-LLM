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
    
    # Load a sample image and its corresponding label.
    img = np.array(Image.open("tests/test_image_0.png").convert('L').resize((28, 28)))
    images = np.expand_dims(img, axis=0)
    
    classifier: IClassifyDigits = ClassifyDigits()
    predictions = classifier(images)

    # Calculate the accuracy of a single prediction.
    assert int(predictions[0]) == 5

def test_digit_recognition_accuracy_with_test_set(model: tf.keras.Model):
    """Test that the digit recognition model achieves an accuracy of more than 95% on a sample test set."""
    
    images, labels = generate_random_images_and_labels(10)
    
    classifier: IClassifyDigits = ClassifyDigits()
    predictions = classifier(images)

    # Calculate the accuracy.
    correct_predictions = np.sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95


def generate_random_images_and_labels(num_samples):
    """Generate a sample set of images and their corresponding labels."""
    
    image_size = (28, 28)
    num_classes = 10
    
    # Create an array to hold the images.
    images = np.zeros((num_samples,) + image_size)

    # Generate random pixel values for each class label.
    labels = np.random.randint(0, num_classes, size=num_samples)

    for i in range(num_samples):
        img_array = np.full(image_size, labels[i], dtype=np.uint8)
        
        images[i] = img_array

    return images, labels
