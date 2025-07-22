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
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match input format expected by classifier
        preprocessed_images = []
        for image in x_test:
            img = Image.fromarray(image, mode='L')
            resized_img = np.array(img.resize((28, 28)))
            preprocessed_images.append(resized_img)
        
        predictions = classifier(np.array(preprocessed_images))
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy >= 0.95

    def test_invalid_input_shape(self, classifier: IClassifyDigits) -> None:
        with pytest.raises(ValueError):
            classifier(np.random.rand(1, 10))

