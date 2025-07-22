import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    images = []
    labels = []
    
    for i in range(num_images):
        img_array = np.random.randint(0, 256, size=(image_size[1], image_size[0]), dtype=np.uint8)
        label = np.random.randint(0, 10)  # Random digit between 0 and 9
        
        images.append(img_array / 255.0)  # Normalize
        labels.append(label)
    
    return np.array(images), np.array(labels)

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    classifier = ClassifyDigits()
    images, expected_labels = test_set
    
    predicted_labels = classifier(np.reshape(images, (-1, 28 * 28)))
    
    # Calculate accuracy
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        predictions = model.predict(images.reshape(-1, 28*28))
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
