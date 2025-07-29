
# tests/test_digit_recognition.py
import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

def test_load_model():
    """Test if model is loaded successfully"""
    assert isinstance(constants.MODEL_DIGIT_RECOGNITION_PATH, str)
    try:
        tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

def test_classify_digits():
    """Test if digits are classified correctly"""
    classifier = ClassifyDigits()
    
    # Create a dataset of single-digit numbers (0 through 9) and other random string data
    images = []
    labels = []
    for i in range(10):
        img = Image.new('L', (28, 28))
        pixels = img.load()
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if (x - 14) ** 2 + (y - 14) ** 2 <= i * 10:
                    pixels[x, y] = 255
        images.append(np.array(img))
        labels.append(i)
    
    # Add some random string data to the dataset
    for _ in range(5):
        img = Image.new('L', (28, 28))
        pixels = img.load()
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if np.random.rand() < 0.5:
                    pixels[x, y] = 255
        images.append(np.array(img))
        labels.append(-1)  # Label -1 indicates non-digit data
    
    # Test the classifier with at least ten different inputs from the dataset
    correct_count = 0
    for i in range(len(images)):
        prediction = classifier([images[i]])
        if (prediction[0] == labels[i]) or (labels[i] == -1 and prediction[0] not in range(10)):  # If non-digit data, any digit is incorrect
            correct_count += 1
    
    accuracy = correct_count / len(images)
    
    assert accuracy > 0.95

def test_classify_digits_empty_input():
    """Test if classifier handles empty input correctly"""
    classifier = ClassifyDigits()
    with pytest.raises(ValueError):
        classifier([])

def test_classify_digits_invalid_image_size():
    """Test if classifier handles invalid image size correctly"""
    classifier = ClassifyDigits()
    
    # Create an image of incorrect size
    img = Image.new('L', (10, 20))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if np.random.rand() < 0.5:
                pixels[x, y] = 255
    
    with pytest.raises(ValueError):
        classifier([np.array(img)])
