
import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Load test images and their corresponding labels
def load_test_images():
    test_dir = 'test_data'
    test_labels = []
    test_images = []

    for filename in os.listdir(test_dir):
        img = Image.open(os.path.join(test_dir, filename)).convert('L').resize((28, 28))
        image_array = np.array(img)
        label = int(filename.split('.')[0])
        
        test_labels.append(label)
        test_images.append(image_array)

    return np.array(test_images), np.array(test_labels)


# Test the ClassifyDigits function
def test_classify_digits():
    # Load model and create an instance of ClassifyDigits class
    classify_digits = ClassifyDigits()

    # Get test images and labels from a directory named 'test_data'
    test_images, test_labels = load_test_images()
    
    predictions = []
    for image in test_images:
        prediction = int(classify_digits(np.array([image]))[0])
        predictions.append(prediction)

    accuracy = np.sum(np.equal(predictions, test_labels)) / len(test_labels)
    assert accuracy > 0.95


# Test the model's performance on a single input
def test_single_input():
    classify_digits = ClassifyDigits()
    
    # Create an image of digit '5'
    img_array = np.zeros((28, 28))
    for i in range(10):
        for j in range(20):
            if (i-4)**2 + (j-14)**2 < 50:
                img_array[i][j] = 255
    
    prediction = int(classify_digits(np.array([img_array]))[0])
    
    assert prediction == 5


# Test the model's performance on an invalid input
def test_invalid_input():
    classify_digits = ClassifyDigits()
    
    # Create a random image that is not a digit
    img_array = np.random.randint(256, size=(28, 28))
    
    with pytest.raises(ValueError):
        prediction = int(classify_digits(np.array([img_array]))[0])
