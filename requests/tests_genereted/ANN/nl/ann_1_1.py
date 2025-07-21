import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image
import pytest

class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        return ClassifyDigits()

    @pytest.mark.parametrize("num_samples", [100, 500])
    def test_classification_accuracy(self, num_samples: int, classifier: IClassifyDigits) -> None:
        # Load MNIST dataset
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a random subset of the training data for testing
        indices = np.random.choice(x_train.shape[0], size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match input format expected by classifier
        test_images: list[Image] = []
        for image in x_test:
            pil_image = Image.fromarray(image, mode='L')
            resized_pil_image = pil_image.resize((28, 28))
            test_images.append(resized_pil_image)

        # Convert images to numpy array and classify
        test_array: np.ndarray = np.array([np.array(img) for img in test_images])
        predictions = classifier(test_array)
        
        # Calculate accuracy of classification model
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy >= 0.95

    def test_invalid_input(self, classifier: IClassifyDigits):
        with pytest.raises(ValueError):
            classifier(np.array([1]))

