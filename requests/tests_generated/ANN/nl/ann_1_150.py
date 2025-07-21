import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

def test_model_accuracy():
    # Load MNIST dataset for testing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess the data to match model input format
    x_test_preprocessed = np.array([np.array(Image.fromarray(image).resize((28, 28)).convert('L')) / 255.0 for image in x_test])

    classifier: IClassifyDigits = ClassifyDigits()
    
    predictions = classifier(x_test_preprocessed)
    
    accuracy = accuracy_score(y_test[:len(predictions)], predictions)

    assert np.round(accuracy * 100) >= 95, f"Model accuracy is {np.round(accuracy * 100)}%, expected at least 95%"
