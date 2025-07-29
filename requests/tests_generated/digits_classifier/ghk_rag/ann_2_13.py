
# tests/test_digit_recognition.py
import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

@pytest.fixture
def model():
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)

@pytest.mark.parametrize("image_path", [
    "path/to/image0.png",
    "path/to/image1.png",
    "path/to/image2.png",
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
        image_path = f"path/to/image{i}.png"
        x = Image.open(image_path).convert('L').resize((28, 28))
        images.append(np.array(x))
        labels.append(i)

    classifier = ClassifyDigits()
    
    predictions = []
    for i in range(len(images)):
        prediction = classifier(images=images[i])
        predictions.append(prediction)
        
    accuracy = np.sum([1 if pred == label else 0 for pred, label in zip(predictions, labels)]) / len(labels) * 100
    
    assert accuracy > 95.0, "Model should recognize over 95% of digits correctly"

def test_digit_recognition_multiple_inputs(model):
    # Load dataset of single-digit numbers (0 through 9)
    images = []
    
    for i in range(10):  # Loop over digits from 0 to 9
        image_path = f"path/to/image{i}.png"
        x = Image.open(image_path).convert('L').resize((28, 28))
        images.append(np.array(x))

    classifier = ClassifyDigits()
    
    predictions = []
    for i in range(len(images)):
        prediction = classifier(images=images[i])
        predictions.append(prediction)
        
    assert len(predictions) >= 10, "Should recognize at least ten different inputs"
