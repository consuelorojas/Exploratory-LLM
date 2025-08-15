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

    assert isinstance(prediction, NDArray)


@pytest.mark.parametrize("image_array", [
    np.random.randint(0, 256, size=(1, 28 * 28), dtype=np.uint8),
    # Add more image arrays here
])
def test_digit_recognition_with_random_images(image_array: NDArray):
    classifier = ClassifyDigits()
    prediction = classifier(images=image_array)

    assert isinstance(prediction, NDArray)


@pytest.mark.parametrize("image_path", [
    "path_to_image_0.png",
    "path_to_image_1.png",
    # Add more image paths here
])
def test_digit_recognition_accuracy(image_path: str):
    classifier = ClassifyDigits()
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)

    assert isinstance(prediction, NDArray)


# Test with a dataset
def test_digit_recognition_with_dataset():
    # Load your dataset here (e.g., from a file or database)
    dataset_images = [
        "path_to_image_0.png",
        "path_to_image_1.png",
        # Add more image paths here
    ]
    correct_count = 0

    classifier = ClassifyDigits()
    for i, image_path in enumerate(dataset_images):
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        prediction = classifier(images=images)

        # Assuming the actual digit is stored somewhere (e.g., a dictionary or another array)
        if int(prediction) == get_actual_digit(i):  # Replace 'get_actual_digit' with your function
            correct_count += 1

    accuracy = correct_count / len(dataset_images)
    assert accuracy > 0.95


def test_classify_digits_interface():
    classifier = ClassifyDigits()
    images: NDArray = np.random.randint(0, 256, size=(10, 28 * 28), dtype=np.uint8)

    predictions = classifier(images=images)
    assert isinstance(predictions, NDArray)


# Replace 'get_actual_digit' with your function to get the actual digit from the dataset
def get_actual_digit(index: int) -> int:
    # Your implementation here
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--durations=10"])
