import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    # For demonstration purposes, we'll use a simple dataset with 10 images.
    # In practice, you would replace this with your actual test data.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Replace with real images
    labels = np.arange(0, num_images % 10)  # Replace with actual labels
    
    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten the input data
            normalized_images = (images / 255.0).reshape(-1, 28 * 28)
            
            predictions = model.predict(normalized_images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    images, labels = test_set
    
    # Classify the test set using the trained model
    classifier = ClassifyDigits()
    predicted_labels = classifier(images=images)
    
    accuracy = sum(predicted_labels == labels) / len(labels)
    
    assert accuracy > 0.95

def main():
    pytest.main([__file__, '-v'])

if __name__ == "__main__":
    main()

