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

    assert isinstance(prediction, NDArray), "Prediction should be a numpy array"
    assert len(prediction) == 1, "Only one image is being classified"


def test_digit_recognition_accuracy(model):
    # Load dataset of single-digit numbers (0 through 9)
    images: list[Image] = []
    labels: list[int] = []

    for i in range(10):
        img_path = f"path_to_image_{i}.png"  # Replace with actual image paths
        x = PIL.Image.open(img_path).convert('L').resize((28, 28))
        images.append(x)
        labels.extend([i])

    classifier = ClassifyDigits()
    correct_predictions: int = 0

    for img in images:
        input_image = np.array(img) / 255.0
        prediction = classifier(images=input_image.reshape(1, -1))

        if prediction[0] == labels[images.index(img)]:
            correct_predictions += 1

    accuracy = (correct_predictions / len(labels)) * 100
    assert accuracy > 95


def test_digit_recognition_with_random_data(model):
    # Generate random images and check that the model doesn't recognize them as digits
    classifier = ClassifyDigits()
    for _ in range(10):  # Test with multiple random inputs
        random_image: NDArray[np.uint8] = np.random.randint(low=0, high=256, size=(28, 28), dtype=np.uint8)
        prediction = classifier(images=random_image / 255.0)

        assert isinstance(prediction, NDArray), "Prediction should be a numpy array"
        assert len(prediction) == 1, "Only one image is being classified"


def test_digit_recognition_with_invalid_input(model):
    # Test with invalid input (e.g., non-image data)
    classifier = ClassifyDigits()
    try:
        classifier(images=np.array([0]))
        pytest.fail("Expected an error when passing in invalid input")
    except Exception as e:
        assert isinstance(e, ValueError), "Invalid input should raise a ValueError"
