
# tests/test_digit_recognition.py

import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

def test_load_model():
    """Test if model is loaded successfully."""
    assert isinstance(constants.MODEL_DIGIT_RECOGNITION_PATH, str)
    try:
        tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

def test_classify_digits():
    """Test if digits are classified correctly."""
    classifier = ClassifyDigits()
    
    # Create a dataset of single-digit images
    digit_images = []
    for i in range(10):
        img = Image.new('L', (28, 28), color=0)
        pixels = img.load()
        
        # Draw the digit on the image
        if i == 1:
            pixels[14, 12] = 255
            pixels[15, 11] = 255
            pixels[16, 10] = 255
            pixels[17, 9] = 255
            pixels[18, 8] = 255
        elif i == 2:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 3:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 4:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 5:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 6:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 7:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 8:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 9:
            pixels[14, 12] = 255
            pixels[13, 11] = 255
            pixels[12, 10] = 255
            pixels[15, 9] = 255
            pixels[16, 8] = 255
        elif i == 0:
            for x in range(28):
                pixels[x, 14] = 255
        
        digit_images.append(np.array(img))
    
    # Test the classifier with at least ten different inputs from dataset
    correct_count = 0
    total_inputs = len(digit_images)
    for i, img in enumerate(digit_images[:10]):
        prediction = classifier([img])
        if int(prediction[0]) == i:
            correct_count += 1
    
    # Check if the accuracy is over 95%
    assert (correct_count / total_inputs) >= 0.95

def test_classify_non_digits():
    """Test if non-digit images are classified incorrectly."""
    classifier = ClassifyDigits()
    
    # Create a dataset of non-digit images
    non_digit_images = []
    for _ in range(10):
        img = Image.new('L', (28, 28), color=255)
        
        # Draw some random noise on the image
        pixels = img.load()
        for x in range(28):
            for y in range(28):
                if np.random.rand() < 0.5:
                    pixels[x, y] = 0
        
        non_digit_images.append(np.array(img))
    
    # Test the classifier with at least ten different inputs from dataset
    correct_count = 0
    total_inputs = len(non_digit_images)
    for img in non_digit_images[:10]:
        prediction = classifier([img])
        if int(prediction[0]) not in range(10):
            correct_count += 1
    
    # Check if the accuracy is over 95%
    assert (correct_count / total_inputs) >= 0.95
