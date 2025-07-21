import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images with known labels.
    num_samples = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_samples, *image_size).astype(np.uint8) / 255.0
    labels = np.random.randint(10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Create an instance of IClassifyDigits to classify digits.
    classifier = ClassifyDigits()
    
    # Get the test set
    images, expected_labels = test_set
    
    # Convert and normalize the images for classification
    normalized_images = np.array([np.array(Image.fromarray(image).convert('L').resize((28, 28))) / 255.0 
                                  if len(np.shape(image)) == 3 else image / 255.0 
                                  for image in images])
    
    # Classify the test set using the model
    predicted_labels = classifier(normalized_images)
    
    # Calculate accuracy by comparing expected and actual labels.
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"

class ClassifyDigits:
    def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
        # Load the model
        loaded_model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
        
        # Normalize and flatten the input data.
        normalized_images = images.reshape(-1, 28 * 28)

        predictions = loaded_model.predict(normalized_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
