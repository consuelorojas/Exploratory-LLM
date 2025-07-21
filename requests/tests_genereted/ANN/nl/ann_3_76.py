import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import PIL.Image
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.classifier import ClassifyDigits
import pytest
import os


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("image_path, expected_label", [
    ("path_to_image_0.png", 0),
    ("path_to_image_1.png", 1),
    # Add more test cases here...
])
def test_classification_accuracy(model: tf.keras.Model, image_path: str, expected_label: int):
    classifier = ClassifyDigits()
    prediction = classifier(np.array([PIL.Image.open(image_path).convert('L').resize((28, 28))]))
    assert np.argmax(prediction) == expected_label


@pytest.mark.parametrize("num_images", [10])
def test_batch_classification_accuracy(model: tf.keras.Model, num_images):
    # Generate a batch of random images with known labels
    images = []
    labels = []
    for i in range(num_images):
        image_path = f"path_to_image_{i}.png"
        label = i % 10
        images.append(PIL.Image.open(image_path).convert('L').resize((28, 28)))
        labels.append(label)

    classifier = ClassifyDigits()
    predictions = classifier(np.array(images))
    accuracy = np.mean(predictions == labels)
    assert accuracy >= 0.95


def test_model_accuracy_on_mnist_dataset(model: tf.keras.Model):
    # Load MNIST dataset
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Preprocess images
    x_train = x_train / 255.0

    classifier = ClassifyDigits()
    predictions = classifier(x_train)
    accuracy = np.mean(predictions == y_train)

    assert accuracy >= 0.95


def test_classifier_interface():
    classifier = ClassifyDigits()
    image_path = "path_to_image_0.png"
    prediction = classifier(np.array([PIL.Image.open(image_path).convert('L').resize((28, 28))]))
    assert isinstance(prediction, np.ndarray)
