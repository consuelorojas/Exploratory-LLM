import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
from tensorflow.keras.models import load_model
from PIL import Image
import pytest
import os


@pytest.fixture
def classify_digits():
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            model = load_model(MODEL_DIGIT_RECOGNITION_PATH)
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    yield ClassifyDigits()


def test_classification_accuracy(classify_digits):
    correct_classifications = 0
    total_images = 0
    
    for filename in os.listdir(TEST_IMAGES_PATH):
        if filename.endswith(".png"):
            image_path = os.path.join(TEST_IMAGES_PATH, filename)
            label = int(filename.split("_")[0])
            
            image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))
            prediction = classify_digits(np.expand_dims(image_array, axis=0))
            
            if prediction[0] == label:
                correct_classifications += 1
            
            total_images += 1
    
    accuracy = (correct_classifications / total_images) * 100
    assert accuracy >= 95.0


def test_classification_accuracy_with_mock_data(classify_digits):
    # Generate mock data with known labels and images
    np.random.seed(42)
    
    num_samples = 10000
    image_size = 28
    
    images = np.random.rand(num_samples, image_size, image_size).astype(np.float32) * 255.0
    labels = np.random.randint(low=0, high=10, size=num_samples)

    # Use the model to make predictions on these mock data points
    predicted_labels = classify_digits(images)
    
    accuracy = (np.sum(predicted_labels == labels)) / num_samples
    
    assert accuracy >= 95.0


def test_classification_accuracy_with_real_data(classify_digits):
    from tensorflow.keras.datasets import mnist

    # Load MNIST dataset for testing the model's performance
    (_, _), (test_images, test_labels) = mnist.load_data()
    
    images_normalized = test_images / 255.0
    
    predicted_labels = classify_digits(images_normalized)
    
    accuracy = np.sum(predicted_labels == test_labels) / len(test_labels)

    assert accuracy >= 95.0
