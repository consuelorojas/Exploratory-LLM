import tensorflow as tf

import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os

# Load MNIST test data for testing purposes
def load_mnist_test_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

@pytest.fixture
def classifier():
    """Create a ClassifyDigits instance"""
    return ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image
    """
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    assert isinstance(prediction[0], int)

def test_recognize_digits_accuracy(classifier):
    """
    Test the accuracy of the digit recognition model
    using MNIST test data.
    """
    x_test, y_test = load_mnist_test_data()
    
    # Preprocess images to match input format expected by classifier
    images = np.array([img.reshape(28 * 28) / 255.0 for img in x_test])
    
    predictions = []
    batch_size = 32
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        prediction_batch = classifier(batch_images)
        predictions.extend(prediction_batch)

    accuracy = np.mean(np.array(predictions) == y_test[:len(predictions)])
    
    assert accuracy > 0.95

def test_recognize_digits_multiple_inputs(classifier, tmp_path):
    """
    Test the digit recognition model with multiple inputs
    from a dataset.
    """
    # Create temporary images for testing purposes
    num_images = 10
    
    correct_count = 0
    
    for i in range(num_images):
        img_array = np.random.randint(255, size=(28, 28))
        
        if i < int(num_images * 0.95):  # Simulate mostly correct inputs
            label = i % 10
            img_array[label*3:label*3+5] = [128]*5
        
        image_path = tmp_path / f"image_{i}.png"
        Image.fromarray(img_array).convert('L').save(image_path)
        
        x = PIL.Image.open(str(image_path)).convert('L')
        images = np.array(x)
        
        prediction = classifier(images=images)[0]
        
        if i < int(num_images * 0.95):
            correct_count += (prediction == label % 10)

    accuracy = correct_count / num_images
    
    assert accuracy > 0.95
