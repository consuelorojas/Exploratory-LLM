import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')





import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

def test_classification_accuracy():
    # Load MNIST dataset for testing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Create an instance of the ClassifyDigits class
    classifier: IClassifyDigits = ClassifyDigits()

    # Preprocess images to match model input shape and normalize pixel values
    test_images = np.array([np.array(Image.fromarray(image).resize((28, 28)).convert('L')) for image in x_test])

    # Make predictions using the classify_digits function
    predicted_labels = classifier(test_images)

    # Calculate accuracy of classification
    accuracy = accuracy_score(y_test, predicted_labels)

    # Assert that the model's accuracy is at least 95%
    assert accuracy >= 0.95

test_classification_accuracy()
