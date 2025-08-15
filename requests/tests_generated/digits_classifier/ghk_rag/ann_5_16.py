import tensorflow as tf

import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os

# Load MNIST test data for testing purposes (you can use any dataset you have)
def load_mnist_test_data():
    # Assuming the MNIST dataset is downloaded and stored in a directory named 'mnist'
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    
    return x_test, y_test

# Create an instance of ClassifyDigits for testing purposes
def create_classify_digits_instance():
    classify_digits = ClassifyDigits()
    return classify_digits

@pytest.fixture
def dataset():
    # Load MNIST test data (or any other single-digit number dataset)
    images, labels = load_mnist_test_data()
    
    yield images, labels

# Test the recognition accuracy of digits using a sample from your dataset
def test_recognition_accuracy(dataset):
    classify_digits_instance = create_classify_digits_instance()
    images, labels = dataset
    
    # Select at least ten different inputs for testing (you can use more if needed)
    num_test_samples = 1000
    indices_to_use = np.random.choice(len(images), size=num_test_samples, replace=False)
    
    test_images = images[indices_to_use]
    expected_labels = labels[indices_to_use]

    # Preprocess the selected input data (resize and normalize) before passing to ClassifyDigits instance
    preprocessed_images = []
    for image in test_images:
        img_pil = Image.fromarray(image)
        resized_img = np.array(img_pil.resize((28, 28)))
        
        normalized_image = resized_img / 255.0
        
        # Reshape the flattened array to match input shape expected by ClassifyDigits instance
        preprocessed_image = normalized_image.reshape(-1, 784)  
        
        preprocessed_images.append(preprocessed_image)
    
    predicted_labels = []
    for image in preprocessed_images:
        prediction = classify_digits_instance(image=np.array([image]))
        predicted_label = int(prediction[0])
        
        predicted_labels.append(predicted_label)

    # Calculate the accuracy of digit recognition
    num_correctly_recognized = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    
    accuracy_percentage = (num_correctly_recognized / len(expected_labels)) * 100
    
    assert accuracy_percentage > 95

# Test that the model is loaded correctly
def test_model_loaded():
    classify_digits_instance = create_classify_digits_instance()
    
    # Check if the model attribute exists and has been successfully loaded from file path specified in constants.py module.
    assert hasattr(classify_digits_instance, 'model')
