import tensorflow as tf

import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

# Load test images and their corresponding labels
def load_test_data():
    test_images = []
    test_labels = []
    
    for i in range(10):
        img_path = f"test_image_{i}.png"
        image = Image.open(img_path).convert('L').resize((28, 28))
        array = np.array(image)
        
        # Append the normalized and flattened image to the list
        test_images.append(array / 255.0)
        test_labels.append(i)  # Assuming each image is named after its digit label
    
    return np.array(test_images), np.array(test_labels)

# Test function for recognizing digits correctly
@pytest.mark.parametrize("test_image, expected_label", zip(*load_test_data()))
def test_digit_recognition(test_image, expected_label):
    classifier = ClassifyDigits()
    
    # Reshape the image to match the model's input shape
    reshaped_image = np.reshape(test_image, (1, 28 * 28))
    
    predicted_label = classifier(reshaped_image)[0]
    
    assert predicted_label == expected_label

# Test function for recognizing over 95% of digits correctly
def test_digit_recognition_accuracy():
    classifier = ClassifyDigits()
    test_images, test_labels = load_test_data()
    
    # Reshape the images to match the model's input shape
    reshaped_images = np.reshape(test_images, (-1, 28 * 28))
    
    predicted_labels = classifier(reshaped_images)
    
    accuracy = sum(predicted_labels == test_labels) / len(test_labels)
    
    assert accuracy > 0.95

# Test function for loading the model from a file
def test_load_model():
    try:
        tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")

