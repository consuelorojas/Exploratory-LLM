import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image
import pytest

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        # implementation of the model prediction
        pass  # Replace with your actual implementation


@pytest.fixture
def classify_digits():
    return ClassifyDigits()


@pytest.mark.parametrize("num_samples", [1000])
def test_classification_accuracy(classify_digits, num_samples):
    (x_train, y_train), (_, _) = mnist.load_data()
    
    # Select a subset of the training data for testing
    indices = np.random.choice(x_train.shape[0], size=num_samples)
    x_test = x_train[indices]
    y_test = y_train[indices]

    # Preprocess images to match model input shape and type
    x_test_preprocessed = []
    for image in x_test:
        img = Image.fromarray(image, mode='L')
        img_resized = np.array(img.resize((28, 28)))
        x_test_preprocessed.append(img_resized)
    
    # Convert list of images to numpy array
    x_test_array = np.stack(x_test_preprocessed)

    predictions = classify_digits(images=x_test_array / 255.0)  
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)

    assert accuracy >= 0.95

