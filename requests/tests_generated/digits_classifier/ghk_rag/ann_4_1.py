
import pytest
from numpy.typing import NDArray
import tensorflow as tf
import interfaces
import PIL.Image
import numpy as np
import constants
from your_module import ClassifyDigits  # Replace 'your_module' with actual module name


@pytest.fixture
def model():
    print("Loading model from:", constants.MODEL_DIGIT_RECOGNITION_PATH)
    model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
    return tf.keras.models.load_model(model_path, compile=False)


class TestClassifyDigits:
    @pytest.mark.parametrize(
        "image_path, expected_digit",
        [
            ("path_to_image_0.png", 0),
            ("path_to_image_1.png", 1),
            ("path_to_image_2.png", 2),
            ("path_to_image_3.png", 3),
            ("path_to_image_4.png", 4),
            ("path_to_image_5.png", 5),
            ("path_to_image_6.png", 6),
            ("path_to_image_7.png", 7),
            ("path_to_image_8.png", 8),
            ("path_to_image_9.png", 9),
        ],
    )
    def test_classify_digits(self, image_path: str, expected_digit: int):
        classify_digits = ClassifyDigits()
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        predicted_digit = classify_digits(images=images)[0]
        assert predicted_digit == expected_digit

    def test_accuracy(self, model: tf.keras.Model):
        # Load dataset of single-digit numbers (0 through 9) and other random string data
        # For this example, we assume the dataset is stored in a list called 'dataset'
        dataset = [
            ("path_to_image_0.png", 0),
            ("path_to_image_1.png", 1),
            ("path_to_image_2.png", 2),
            ("path_to_image_3.png", 3),
            ("path_to_image_4.png", 4),
            ("path_to_image_5.png", 5),
            ("path_to_image_6.png", 6),
            ("path_to_image_7.png", 7),
            ("path_to_image_8.png", 8),
            ("path_to_image_9.png", 9),
        ]

        classify_digits = ClassifyDigits()
        correct_predictions = 0
        total_images = len(dataset)

        for image_path, expected_digit in dataset:
            x = PIL.Image.open(image_path).convert('L').resize((28, 28))
            images = np.array(x)
            predicted_digit = classify_digits(images=images)[0]
            if predicted_digit == expected_digit:
                correct_predictions += 1

        accuracy = (correct_predictions / total_images) * 100
        assert accuracy > 95


def test_classify_digits_interface():
    classify_digits = ClassifyDigits()
    x = PIL.Image.open("path_to_image_0.png").convert('L').resize((28, 28))
    images = np.array(x)
    predicted_digit = classify_digits(images=images)[0]
    assert isinstance(predicted_digit, int)


def test_classify_digits_input_type():
    classify_digits = ClassifyDigits()
    x = PIL.Image.open("path_to_image_0.png").convert('L').resize((28, 28))
    images = np.array(x)
    with pytest.raises(TypeError):
        classify_digits(images="invalid input")


def test_classify_digits_output_type():
    classify_digits = ClassifyDigits()
    x = PIL.Image.open("path_to_image_0.png").convert('L').resize((28, 28))
    images = np.array(x)
    predicted_digit = classify_digits(images=images)[0]
    assert isinstance(predicted_digit, int)


def test_classify_digits_empty_input():
    classify_digits = ClassifyDigits()
    with pytest.raises(ValueError):
        classify_digits(images=np.empty(0))

