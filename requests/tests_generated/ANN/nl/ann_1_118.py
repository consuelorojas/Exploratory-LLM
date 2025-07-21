import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
import PIL.Image
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def classify_digits():
    model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    yield ClassifyDigits()


def test_classification_accuracy(classify_digits):
    # Load the test dataset (assuming it's a set of images with their corresponding labels)
    test_images = []
    test_labels = []
    for i in range(10):  # assuming we have 10 classes
        image_path = f"{TEST_IMAGES_PATH}/class_{i}.png"
        label = i
        image = np.array(PIL.Image.open(image_path).convert('L').resize((28, 28)))
        test_images.append(image)
        test_labels.append(label)

    # Make predictions on the test dataset
    predicted_classes = classify_digits(np.array(test_images))

    # Calculate accuracy using sklearn's accuracy_score function
    accuracy = accuracy_score(test_labels, predicted_classes)

    assert np.round(accuracy * 100) >= 95


def test_classification_accuracy_with_random_data(classify_digits):
    # Generate random images and labels for testing (this is not ideal but can be used if no real data)
    num_samples = 1000
    test_images = np.random.rand(num_samples, 28, 28)
    test_labels = np.random.randint(10, size=num_samples)

    # Make predictions on the generated dataset
    predicted_classes = classify_digits(test_images)

    # Calculate accuracy using sklearn's accuracy_score function (this will likely be low due to random data)
    accuracy = accuracy_score(test_labels, predicted_classes)

    assert np.round(accuracy * 100) >= 95


def test_classification_accuracy_with_mnist_data(classify_digits):
    import tensorflow_datasets as tfds

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Preprocess images to match model input shape and normalize pixel values
    test_images = np.reshape(test_images, (-1, 28, 28))
    
    predicted_classes = classify_digits(test_images)

    accuracy = accuracy_score(np.argmax(predicted_classes[:, None], axis=0), test_labels)
    assert np.round(accuracy * 100) >= 95

