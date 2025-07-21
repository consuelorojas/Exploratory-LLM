import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
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
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Random pixel values between 0 and 255
    
    # Assign some dummy labels for demonstration purposes. In a real scenario,
    # these would be the actual expected classifications.
    labels = np.arange(10) % num_images
    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test that the digit recognition model achieves an accuracy of more than 95%."""
    
    classifier = ClassifyDigits()
    images, expected_labels = test_set
    
    # Convert to grayscale and resize if necessary
    resized_images = np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images])
    
    predicted_labels = classifier(resized_images)
    
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95
```

```python
# tests/conftest.py

import pytest
from digits_classifier.interfaces import IClassifyDigits, ClassifyDigits

@pytest.fixture(autouse=True)
def setup():
    """Setup and teardown for the test environment."""
    # Any global setup or teardown can go here.
    pass
