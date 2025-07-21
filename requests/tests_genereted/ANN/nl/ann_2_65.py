import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import PIL.Image
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.classifier import ClassifyDigits

class TestDigitClassifier:
    def test_classification_accuracy(self):
        # Load the model and classifier
        model = load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
        classifier: IClassifyDigits = ClassifyDigits()

        # Generate a set of random images with known labels (for testing purposes only)
        num_images = 1000
        test_labels = np.random.randint(0, 10, size=num_images)

        # Create dummy images for each label (this should be replaced with actual MNIST data or similar)
        test_images = []
        for i in range(num_images):
            img = PIL.Image.new('L', (28, 28))
            pixels = img.load()
            for x in range(28):
                for y in range(28):
                    # Simple example: set pixel value based on label
                    if test_labels[i] == int((x + y) % 10):  
                        pixels[x, y] = 255
                    else:
                        pixels[x, y] = 0

            img = np.array(img.resize((28, 28)))
            test_images.append(img)

        # Classify the images and calculate accuracy
        predictions = classifier(np.array(test_images))
        correct_predictions = sum(1 for pred, label in zip(predictions, test_labels) if pred == label)
        accuracy = (correct_predictions / num_images) * 100

        assert accuracy >= 95.0, f"Model classification accuracy is {accuracy:.2f}%, which is less than the required 95%"

    def test_classifier_interface(self):
        classifier: IClassifyDigits = ClassifyDigits()
        images = np.random.rand(10, 28 * 28)
        predictions = classifier(images)

        assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
        assert len(predictions) == len(images), "Number of predictions does not match the number of input images"

# Run tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
