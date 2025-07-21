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
        
        # Select a subset of the training data for testing
        indices = np.random.choice(x_train.shape[0], size=num_samples)
        x_test_subset = x_train[indices]
        y_test_subset = y_train[indices]

        # Preprocess images to match input format expected by classifier
        preprocessed_images = []
        for image in x_test_subset:
            pil_image: Image = PIL.Image.fromarray(image, mode='L')
            resized_pil_image: Image = pil_image.resize((28, 28))
            numpy_array = np.array(resized_pil_image)
            preprocessed_images.append(numpy_array)

        # Convert list of images to NumPy array
        x_test_subset_preprocessed = np.stack(preprocessed_images)

        # Classify digits using the classifier under test
        predicted_labels: np.ndarray[np.int_] = classifier(x_test_subset_preprocessed)

        # Calculate accuracy score
        accuracy: float = accuracy_score(y_true=y_test_subset, y_pred=predicted_labels)
        
        assert accuracy >= 0.95

