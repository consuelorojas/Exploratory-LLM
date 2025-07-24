import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')





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
            pil_image: Image = PIL.Image.fromarray(image, 'L')
            resized_pil_image: Image = pil_image.resize((28, 28))
            numpy_array = np.array(resized_pil_image)
            preprocessed_images.append(numpy_array)

        # Make predictions
        y_pred = classifier(np.array(preprocessed_images))

        # Calculate accuracy
        accuracy = accuracy_score(y_test_subset, y_pred)

        assert accuracy >= 0.95

    def test_invalid_input_shape(self, classifier: IClassifyDigits) -> None:
        with pytest.raises(ValueError):
            classifier(np.random.rand(1, 10))  # Invalid shape (should be (n_samples, 28*28))

