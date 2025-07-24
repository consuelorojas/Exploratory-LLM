import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')





import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

def test_model_accuracy():
    # Load MNIST dataset for testing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Create an instance of the ClassifyDigits class
    classifier: IClassifyDigits = ClassifyDigits()

    # Preprocess images to match model input shape and normalize pixel values
    x_test_preprocessed = np.array([np.array(Image.fromarray(image).resize((28, 28)).convert('L')) for image in x_test])

    # Make predictions using the classifier
    y_pred = classifier(x_test_preprocessed)

    # Calculate accuracy of the model on test data
    accuracy = accuracy_score(y_test[:len(y_pred)], y_pred)

    # Assert that the classification accuracy is at least 95%
    assert accuracy >= 0.95, f"Model accuracy {accuracy:.2f} is less than expected (>= 0.95)"
