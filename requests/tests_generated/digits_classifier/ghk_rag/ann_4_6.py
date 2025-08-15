
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
    # Replace 'test_images' with actual path to your test image dataset
    import os
    test_image_paths = [os.path.join('test_images', file) for file in os.listdir('test_images')]
    
    return [(PIL.Image.open(path).convert('L').resize((28, 28)), int(os.path.basename(path)[0])) 
            for path in test_image_paths]

@pytest.fixture
def classify_digits():
    """Create an instance of the ClassifyDigits class."""
    yield ClassifyDigits()

class TestClassifyDigits:
    def test_recognize_more_than_95_percent_correctly(self, classify_digits):
        # Generate or load your dataset containing various single-digit numbers (0 through 9)
        images_and_labels = generate_test_data()
        
        correct_count = 0
        total_images = len(images_and_labels)

        for image, label in images_and_labels:
            np_image = np.array(image) / 255.0
            prediction = classify_digits(np.expand_dims(np_image.reshape(-1), axis=0))
            
            if int(prediction[0]) == label:
                correct_count += 1
        
        accuracy = (correct_count / total_images) * 100

        assert accuracy > 95, f"Expected recognition rate to be over 95%, but got {accuracy}%"

    def test_classify_digits_interface(self):
        """Test that the ClassifyDigits class implements the IClassifyDigits interface."""
        classify_digits = ClassifyDigits()
        
        # Check if __call__ method is implemented
        assert hasattr(classify_digits, '__call__'), "IClassifyDigits must implement the __call__ method"

    def test_classify_digits_input_type(self):
        """Test that the input to ClassifyDigits.__call__ should be a numpy array."""
        classify_digits = ClassifyDigits()
        
        # Test with valid input
        np_image = np.array(PIL.Image.new('L', (28, 28)))
        assert isinstance(classify_digits(np_image), NDArray)

    def test_classify_digits_output_type(self):
        """Test that the output of ClassifyDigits.__call__ should be a numpy array."""
        classify_digits = ClassifyDigits()
        
        # Test with valid input
        np_image = np.array(PIL.Image.new('L', (28, 28)))
        assert isinstance(classify_digits(np_image), NDArray)
