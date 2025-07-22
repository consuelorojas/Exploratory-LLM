import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("test_image_path, expected_label", [
    ("path_to_test_image_1.png", 5),
    ("path_to_test_image_2.png", 3),
    # Add more test images and labels here
])
def test_classification_accuracy(model, test_image_path, expected_label):
    image = np.array(Image.open(test_image_path).convert('L').resize((28, 28)))
    normalized_image = image / 255.0
    flattened_image = normalized_image.reshape(1, -1)
    
    prediction = model.predict(flattened_image)
    predicted_label = int(np.argmax(prediction))
    
    assert predicted_label == expected_label


def test_model_accuracy(model):
    # Load MNIST dataset for testing (you can use any other method to load your data)
    from tensorflow.keras.datasets import mnist
    (_, _), (test_images, test_labels) = mnist.load_data()
    
    # Preprocess the images
    normalized_test_images = test_images / 255.0
    flattened_test_images = normalized_test_images.reshape(-1, 28 * 28)

    predictions = model.predict(flattened_test_images)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    assert accuracy >= 0.95
