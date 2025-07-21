import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images for classification."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to generate or load multiple images.
    img_path = os.path.join(os.getcwd(), "tests", "test_image.png")
    return np.array(open(img_path).convert('L').resize((28, 28)))


@pytest.fixture
def classifier():
    """Instantiates the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Tests that the trained model achieves an accuracy of more than 95% on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (np.ndarray): A sample image for classification.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Generate predictions using the classifier
    prediction = classifier(test_set)

    # For demonstration purposes, we'll assume a correct label is 5.
    # In a real-world scenario, you'd want to use actual labels for your test set.
    expected_label = np.array([5])

    # Calculate accuracy (in this case, it's just one image)
    accuracy = int(prediction == expected_label)

    assert accuracy > 0.95
```

Please note that the above code assumes a few things:

1. You have an actual `test_image.png` in your tests directory.
2. The model is correctly trained and loaded from the specified path (`MODEL_DIGIT_RECOGNITION_PATH`).
3. Your classifier class has been implemented to work with single images (not batches).

In real-world scenarios, you'd want to use a larger test set with actual labels for accurate accuracy calculations.

Also note that this code does not actually calculate an "accuracy" in the classical sense because it only uses one image and its corresponding label. For calculating true accuracy, consider using `tf.keras.metrics.Accuracy` or similar metrics from other libraries like scikit-learn (`sklearn.metrics.accuracy_score`).