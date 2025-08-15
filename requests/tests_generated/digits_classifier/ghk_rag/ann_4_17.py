
import pytest
from numpy.typing import NDArray
import tensorflow as tf
import interfaces
import PIL.Image
import numpy as np
import constants
from your_module import ClassifyDigits  # Replace 'your_module' with actual module name

# Load the model only once for all tests
model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
model = tf.keras.models.load_model(model_path, compile=False)

def generate_test_data():
    """Generate test data by loading images from a directory."""
    # Replace 'test_images' with actual path to your dataset
    import os
    image_paths = [os.path.join('test_images', file) for file in os.listdir('test_images')]
    
    return [(PIL.Image.open(path).convert('L').resize((28, 28)), int(os.path.basename(path)[0])) 
            if len(os.path.basename(path)) > 1 else (None, None) for path in image_paths]

def test_recognize_digits():
    """Test the digit recognition function with at least ten different inputs."""
    
    # Generate test data
    images_with_labels = generate_test_data()
    valid_images_and_labels = [(image, label) for image, label in images_with_labels if image is not None]
    
    classifier = ClassifyDigits()
    correct_count = 0
    
    for i, (x, expected_label) in enumerate(valid_images_and_labels):
        # Convert the PIL Image to a numpy array
        x_array = np.array(x)
        
        predicted_digit = int(classifier(images=x_array)[0])
        
        if predicted_digit == expected_label:
            correct_count += 1
    
    accuracy = (correct_count / len(valid_images_with_labels)) * 100
    
    assert accuracy > 95, f"Accuracy {accuracy} is less than the required threshold of 95%"

def test_recognize_digits_empty_input():
    """Test that an empty input raises a ValueError."""
    
    classifier = ClassifyDigits()
    
    with pytest.raises(ValueError):
        classifier(images=np.array([]))

@pytest.mark.parametrize("image_path, expected_digit", [
    ("path_to_image_0.png", 0),
    ("path_to_image_1.png", 1),
    # Add more test cases here
])
def test_recognize_digits_individual(image_path: str, expected_digit: int):
    """Test the digit recognition function with individual images."""
    
    classifier = ClassifyDigits()
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    predicted_digit = int(classifier(images=np.array(x))[0])
    
    assert predicted_digit == expected_digit

def test_recognize_digits_multiple_inputs():
    """Test the digit recognition function with multiple inputs."""
    
    classifier = ClassifyDigits()
    image_paths = ["path_to_image_0.png", "path_to_image_1.png"]
    images = [np.array(PIL.Image.open(path).convert('L').resize((28, 28))) for path in image_paths]
    
    predicted_digits = classifier(images=np.stack(images))
    
    assert len(predicted_digits) == len(image_paths)
