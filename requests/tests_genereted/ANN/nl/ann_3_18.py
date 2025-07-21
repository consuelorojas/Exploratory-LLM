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
from io import BytesIO


class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        from digits_classifier.classifier import ClassifyDigits  # noqa: F811
        return ClassifyDigits()

    @pytest.mark.parametrize("num_samples", [100, 500])
    def test_classification_accuracy(self, num_samples, classifier):
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a random subset of the training data for testing
        indices = np.random.choice(len(x_train), size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Convert images to grayscale and resize them to 28x28 pixels
        test_images = []
        for image in x_test:
            img = Image.fromarray(image).convert('L')
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            test_image = np.array(Image.open(BytesIO(buffer.getvalue())).resize((28, 28)))
            test_images.append(test_image)
        
        # Convert list of images to numpy array
        x_test_array = np.stack([np.array(image) for image in test_images])

        predictions = classifier(x_test_array)

        accuracy = accuracy_score(y_test, predictions)

        assert accuracy >= 0.95

    def test_invalid_input(self, classifier):
        with pytest.raises(ValueError):
            classifier(np.random.rand(1, 3, 28))

