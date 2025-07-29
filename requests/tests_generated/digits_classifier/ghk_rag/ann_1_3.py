
import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Load test data (MNIST)
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

def create_test_image(label: int) -> str:
    """Create a temporary image file for testing."""
    img_path = f"test_{label}.png"
    img_array = x_train[label]
    img = Image.fromarray(img_array)
    img.save(img_path, "PNG")
    
    return img_path

def test_recognize_digits():
    # Initialize the classifier
    classify_digits = ClassifyDigits()
    
    correct_count = 0
    
    for i in range(10):
        label = y_train[i]
        
        # Create a temporary image file
        img_path = create_test_image(i)
        
        try:
            x = Image.open(img_path).convert('L').resize((28, 28))
            images = np.array(x)
            
            prediction = classify_digits(images=images)[0]
            if label == prediction:
                correct_count += 1
                
        finally:
            # Remove the temporary image file
            os.remove(img_path)

    accuracy = (correct_count / 10) * 100
    
    assert accuracy > 95

def test_recognize_multiple_images():
    classify_digits = ClassifyDigits()
    
    images_list = []
    labels_list = []
    
    for i in range(20):
        label = y_train[i]
        
        x = Image.open(create_test_image(i)).convert('L').resize((28, 28))
        img_array = np.array(x)
        os.remove(f"test_{i}.png")
        
        images_list.append(img_array)
        labels_list.append(label)

    # Stack the image arrays
    stacked_images = np.stack(images_list)
    
    predictions = classify_digits(stacked_images)
    
    correct_count = sum(1 for label, prediction in zip(labels_list, predictions) if label == prediction)
    
    accuracy = (correct_count / len(predictions)) * 100
    
    assert accuracy > 95

def test_invalid_input():
    with pytest.raises(ValueError):
        ClassifyDigits()(images=np.array([[[[0]]]]))

# Test the model loading
@pytest.fixture(autouse=True)
def check_model_loading():
    try:
        yield
    finally:
        # Check if the model was loaded correctly
        assert os.path.exists(model_path)

