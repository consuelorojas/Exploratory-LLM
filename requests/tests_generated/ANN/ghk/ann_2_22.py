import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


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
    
    # Create dummy data for testing. Replace this with actual test data in your project structure.
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        
    labels = np.random.randint(0, 9, num_images)  # Randomly generate labels
    images = []
    for i in range(num_images):
        img_path = f'test_data/image_{i}.png'
        image_array = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        
        if not os.path.exists(img_path):  # Create a dummy image
            Image.fromarray(image_array).save(img_path, 'PNG')
            
        images.append(np.array(Image.open(img_path).convert('L').resize(image_size)))
    
    return np.array(images), labels

def test_digit_recognition_accuracy(model: IClassifyDigits):
    """Test the accuracy of digit recognition."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten images
            normalized_images = images / 255.0
            flattened_images = normalized_images.reshape(-1, 28 * 28)
            
            predictions = model.predict(flattened_images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    classifier = ClassifyDigits()
    test_set_images, labels = test_set
    
    # Get predicted labels
    predicted_labels = classifier(test_set_images)
    
    accuracy = sum(predicted_labels == labels) / len(labels)
    assert accuracy > 0.95

def main():
    pytest.main([__file__, '-v'])

if __name__ == "__main__":
    main()
