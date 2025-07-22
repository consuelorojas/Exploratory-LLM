import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py
import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate some random images (this should be replaced with your actual test data)
    num_images = 1000
    image_size = 28 * 28
    images = np.random.rand(num_images, image_size).astype(np.float32) / 255.0
    
    # Assign labels to the generated images (again, this is just for demonstration purposes)
    labels = np.random.randint(10, size=num_images)
    
    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test that the digit recognition model achieves an accuracy of more than 95%."""
    # Extract the images and their corresponding labels from the test set
    images, labels = test_set
    
    # Make predictions using the loaded model
    predictions = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in images])
    
    # Calculate accuracy by comparing predicted values with actual labels
    accuracy = sum(predictions == labels) / len(labels)
    
    assert accuracy > 0.95


def test_classify_digits_interface():
    """Test the IClassifyDigits interface."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # For demonstration purposes, assume we have a model that always predicts correctly
            return np.array([int(np.argmax(image)) for image in images])
    
    classifier = ClassifyDigits()
    test_images = np.random.rand(10, 28 * 28).astype(np.float32)
    predictions = classifier(test_images)
    
    assert isinstance(predictions, np.ndarray)


def test_load_image_and_classify():
    """Test loading an image and classifying it using the IClassifyDigits interface."""
    from digits_classifier import ClassifyDigits
    
    # Load a sample image (replace with your actual image path)
    image_path = "path_to_your_test_image.png"
    
    classifier = ClassifyDigits()
    test_images = np.array([np.array(Image.open(image_path).convert('L').resize((28, 28)))])
    
    prediction = classifier(test_images)[0]
    
    assert isinstance(prediction, int)
