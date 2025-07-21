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

@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual dataset here.
    test_set_images = np.random.rand(num_images, *image_size).astype(np.uint8)
    test_set_labels = np.random.randint(0, 10, num_images)

    return test_set_images, test_set_labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Extract images and labels from the test set
    test_set_images, expected_labels = test_set
    
    # Create an instance of IClassifyDigits to classify digits using our loaded model.
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            """Normalizes and flattens the input images before making predictions."""
            
            normalized_images = images / 255.0
            flattened_images = normalized_images.reshape(-1, 28 * 28)
            
            # Use our loaded model to make predictions on these test set images.
            predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])
            
            return predicted_labels
    
    classifier: IClassifyDigits = ClassifyDigits()
    
    # Convert the PIL Image data into a numpy array
    image_array = [np.array(Image.fromarray(image).convert('L').resize((28, 28))) for image in test_set_images]
    
    # Make predictions on our test set using this classifier.
    predicted_labels = classifier(np.stack(image_array))
    
    # Calculate the accuracy of these predictions against the expected labels.
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(test_set[1])) * 100
    
    assert accuracy > 95, f"Expected an accuracy greater than 95%, but got {accuracy:.2f}%"
