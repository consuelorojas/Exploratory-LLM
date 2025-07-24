import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

class TestDigitClassification:
    def test_classification_accuracy(self):
        # Load MNIST dataset for testing
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape and normalize the data
        x_test = np.array([np.array(Image.fromarray(image).resize((28, 28)).convert('L')) for image in x_test])
        
        classifier: IClassifyDigits = ClassifyDigits()
        
        predictions = classifier(x_test)
        
        accuracy = accuracy_score(y_test, predictions)

        # Assert that the classification accuracy is at least 95%
        assert accuracy >= 0.95
