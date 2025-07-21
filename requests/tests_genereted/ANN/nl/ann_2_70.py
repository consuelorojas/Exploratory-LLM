import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_DATA_PATH
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import pytest


@pytest.fixture
def model():
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("image_path", [os.path.join(TEST_DATA_PATH, f"test_image_{i}.png") for i in range(10)])
def test_classification_accuracy(model, image_path):
    # Load the test data
    images = []
    labels = []

    for file_name in os.listdir(TEST_DATA_PATH):
        if not file_name.startswith("test_label_"):
            continue

        label_file_path = os.path.join(TEST_DATA_PATH, file_name)
        with open(label_file_path) as f:
            label = int(f.read())

        image_file_path = os.path.join(TEST_DATA_PATH, file_name.replace("label", "image"))
        images.append(np.array(Image.open(image_file_path).convert('L').resize((28, 28))))
        labels.append(label)

    # Normalize and flatten the test data
    images = np.array(images) / 255.0
    images = images.reshape(-1, 28 * 28)
    
    predictions = model.predict(images)
    predicted_labels = [int(np.argmax(prediction)) for prediction in predictions]

    accuracy = accuracy_score(labels, predicted_labels)

    assert accuracy >= 0.95


def test_model_output_shape(model):
    # Generate a random input
    images = np.random.rand(1, 28 * 28)

    output = model.predict(images)
    
    assert len(output.shape) == 2 and output.shape[0] == 1 and output.shape[1] == 10


def test_model_output_type(model):
    # Generate a random input
    images = np.random.rand(1, 28 * 28)

    output = model.predict(images)
    
    assert isinstance(output, np.ndarray) or (hasattr(output, 'numpy') and hasattr(output.numpy(), '__call__'))
