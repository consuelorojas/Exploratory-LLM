import tensorflow as tf

# tests/test_digit_recognition.py
import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

@pytest.fixture
def model():
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    "path/to/image3.png",
    # Add more image paths here...
])
def test_digit_recognition(model, image_path):
    x = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    
    classifier = ClassifyDigits()
    prediction = classifier(images=images)

    assert isinstance(prediction, int), "Prediction should be an integer"

def test_digit_recognition_accuracy(model):
    # Load dataset of single-digit numbers (0 through 9) and other random string data
    images = []
    labels = []

    for i in range(10):  # Loop over digits from 0 to 9
        image_path = f"path/to/digit_{i}.png"
        x = Image.open(image_path).convert('L').resize((28, 28))
        images.append(np.array(x))
        labels.append(i)

    classifier = ClassifyDigits()
    
    predictions = []
    for image in images:
        prediction = classifier(images=image)
        predictions.extend(prediction)  # Extend list with predicted digit

    accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
    assert accuracy > 0.95, "Digit recognition accuracy should be over 95%"

def test_digit_recognition_multiple_inputs(model):
    images = []
    
    # Load dataset of single-digit numbers (0 through 9) and other random string data
    for i in range(10):  
        image_path = f"path/to/digit_{i}.png"
        x = Image.open(image_path).convert('L').resize((28, 28))
        images.append(np.array(x))

    classifier = ClassifyDigits()
    
    predictions = []
    for image in images:
        prediction = classifier(images=image)
        predictions.extend(prediction)  

    accuracy = sum(1 for pred, label in zip(predictions, range(10)) if pred == label) / len(range(10))
    assert accuracy > 0.95, "Digit recognition accuracy should be over 95%"

def test_digit_recognition_invalid_input(model):
    classifier = ClassifyDigits()
    
    # Test with invalid input (e.g., non-image data)
    images = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        classifier(images=images)

# Add more tests as needed...
