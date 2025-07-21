import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_DATA_DIR
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import pytest


@pytest.fixture
def model():
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("image_path", [os.path.join(TEST_DATA_DIR, f"test_image_{i}.png") for i in range(10)])
def test_classification_accuracy(model, image_path):
    # Load the test data
    images = []
    labels = []
    for file_name in os.listdir(TEST_DATA_DIR):
        if file_name.endswith(".png"):
            img = np.array(Image.open(os.path.join(TEST_DATA_DIR, file_name)).convert('L').resize((28, 28)))
            label = int(file_name.split("_")[2].split('.')[0])
            images.append(img)
            labels.append(label)

    # Normalize and flatten the test data
    images = np.array(images) / 255.0
    images = images.reshape(-1, 28 * 28)

    # Make predictions on the test data
    predictions = model.predict(images)

    # Get the predicted classes
    predicted_classes = [int(np.argmax(prediction)) for prediction in predictions]

    # Calculate the accuracy of the model
    accuracy = accuracy_score(labels, predicted_classes)
    
    assert accuracy >= 0.95


def test_model_loads_correctly():
    try:
        load_model(MODEL_DIGIT_RECOGNITION_PATH)
    except OSError as e:
        pytest.fail(f"Model failed to load: {e}")
