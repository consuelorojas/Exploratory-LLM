import tensorflow as tf

import pytest
from numpy.typing import NDArray
import numpy as np
from PIL.Image import Image
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name


@pytest.fixture
def model():
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)


@pytest.mark.parametrize("image_path", [
    "path_to_image_0.png",
    "path_to_image_1.png",
    "path_to_image_2.png",
    # Add more image paths here
])
def test_digit_recognition(image_path: str):
    classifier = ClassifyDigits()
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)

    assert isinstance(prediction, NDArray) and len(prediction.shape) == 1


def test_digit_recognition_accuracy(model):
    # Load dataset
    dataset_images: list[Image] = []
    dataset_labels: list[int] = []

    for i in range(10):  # Assuming you have images of digits from 0 to 9
        image_path = f"path_to_image_{i}.png"
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        dataset_images.append(x)
        dataset_labels.extend([i] * 10)  # Assuming each digit has 10 images

    # Add random string data to the dataset
    for _ in range(100):
        image = PIL.Image.new('L', (28, 28), color=255)
        dataset_images.append(image)
        dataset_labels.append(-1)  # Labeling non-digit as -1

    # Convert images and labels to numpy arrays
    dataset_images_array: NDArray[np.uint8] = np.array([np.array(x) for x in dataset_images])
    dataset_labels_array: NDArray[int] = np.array(dataset_labels)

    classifier = ClassifyDigits()
    predictions = []
    for image in dataset_images:
        images = np.array(image)
        prediction = classifier(images=images)[0]
        predictions.append(prediction)

    # Calculate accuracy
    correct_predictions = sum(1 for label, pred in zip(dataset_labels_array, predictions) if (label == -1 and pred not in range(10)) or (label != -1 and label == pred))
    total_images = len(predictions)
    accuracy = correct_predictions / total_images

    assert accuracy > 0.95
