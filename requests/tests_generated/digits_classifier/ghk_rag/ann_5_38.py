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

class TestDigitRecognition:
    @pytest.fixture
    def classifier(self):
        return ClassifyDigits()

    @pytest.fixture
    def test_images_and_labels(self):
        images, labels = load_mnist_test_data()
        # Normalize and flatten the images for testing purposes
        normalized_images = (images / 255.0).reshape(-1, 28 * 28)
        return normalized_images[:10], labels[:10]

    @pytest.fixture
    def single_image(self):
        image_path = "path_to_your_test_image.png"
        if not os.path.exists(image_path):
            pytest.skip("Test image does not exist")
        
        x = Image.open(image_path).convert('L').resize((28, 28))
        return np.array(x)

    @pytest.mark.parametrize(
        "image",
        [
            # Add your test images here
            # For example:
            # "path_to_your_test_image_1.png",
            # "path_to_your_test_image_2.png"
        ]
    )
    def test_single_digit_recognition(self, classifier: ClassifyDigits, image):
        if not os.path.exists(image):
            pytest.skip("Test image does not exist")
        
        x = Image.open(image).convert('L').resize((28, 28))
        predicted_label = classifier(np.array(x))[0]
        # Assuming the test images are correctly labeled
        expected_label = int(os.path.basename(image)[0])
        assert predicted_label == expected_label

    def test_digit_recognition_accuracy(self, classifier: ClassifyDigits, test_images_and_labels):
        test_images, labels = test_images_and_labels
        predictions = [int(np.argmax(classifier(images.reshape(1, -1)))) for images in test_images]
        accuracy = sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
        assert accuracy > 0.95

    def test_single_digit_recognition_from_image(self, classifier: ClassifyDigits, single_image):
        predicted_label = classifier(single_image)[0]
        # Assuming the test image is correctly labeled
        expected_label = int(os.path.basename("path_to_your_test_image.png")[0])
        assert predicted_label == expected_label

    def test_model_loading(self):
        try:
            tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

