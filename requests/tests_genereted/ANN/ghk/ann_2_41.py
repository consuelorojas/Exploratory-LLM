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
        image_array = np.zeros((28, 28), dtype=np.uint8)
        
        if not os.path.exists(img_path):  
            Image.fromarray(image_array).save(img_path)

        images.append(np.array(Image.open(img_path).convert('L').resize(image_size)))
    
    return np.stack(images), labels

def test_digit_recognition_accuracy(model, test_set):
    """Tests the accuracy of digit recognition using a trained model."""
    classifier = ClassifyDigits()
    images, expected_labels = test_set
    
    # Normalize and flatten images
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)
    
    predicted_labels = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in flattened_images])
    
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95
```

```python
# tests/conftest.py

import pytest
from digits_classifier.interfaces import IClassifyDigits

@pytest.fixture(autouse=True, scope="session")
def setup():
    """Setup and teardown for the test session."""
    # Any global setup or teardown code can go here.
    pass
