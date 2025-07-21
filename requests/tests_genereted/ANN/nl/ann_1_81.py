import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL import Image
import pytest

class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        return ClassifyDigits()

    @pytest.mark.parametrize("num_samples", [100, 500])
    def test_classification_accuracy(self, num_samples: int, classifier: IClassifyDigits) -> None:
        # Load MNIST dataset
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a subset of the training data for testing
        indices = np.random.choice(len(x_train), size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match input format expected by classifier
        test_images = []
        for image in x_test:
            pil_image = Image.fromarray(image).resize((28, 28))
            test_images.append(np.array(pil_image))

        predictions = classifier(images=np.array(test_images))
        
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy >= 0.95
