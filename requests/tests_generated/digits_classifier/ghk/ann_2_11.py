import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image
import os


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
    
    # Create dummy data for testing. Replace this with actual test data in your project structure.
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        
    labels = np.random.randint(0, 9, num_images)  # Random digit labels
    images = []
    for i in range(num_images):
        img_path = f'test_data/image_{i}.png'
        image_array = np.random.rand(*image_size)
        Image.fromarray((255 * image_array).astype(np.uint8)).convert('L').save(img_path, 'PNG')
        
        # Load the saved images
        loaded_image = open(img_path).resize(image_size).convert('L')
        images.append(loaded_image)

    return np.array([np.array(im) for im in images]), labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests that the digit recognition model achieves an accuracy of more than 95% on a given test set."""
    
    # Create an instance of ClassifyDigits
    classifier = IClassifyDigits()
    
    # Get images and labels from the test set fixture
    images, expected_labels = test_set
    
    # Use the loaded model to classify the digits in the test set
    predicted_labels = np.array([int(np.argmax(model.predict(im.reshape(-1, 28 * 28) / 255.0))) for im in images])
    
    # Calculate accuracy by comparing predicted labels with expected labels
    correct_predictions = sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95


def test_classify_digits_interface():
    """Tests the ClassifyDigits interface."""
    class MockClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            return np.array([1] * len(images))
    
    classifier = MockClassifyDigits()
    mock_images = np.random.rand(10, 28*28)
    result = classifier(mock_images)
    assert isinstance(result, np.ndarray)


def test_model_load():
    """Tests that the model can be loaded."""
    try:
        load_model(MODEL_DIGIT_RECOGNITION_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

print("done running script")