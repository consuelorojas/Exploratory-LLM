import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
import PIL.Image
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def classify_digits():
    model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    yield ClassifyDigits()


def test_classification_accuracy(classify_digits):
    # Load the test dataset (assuming it's stored as a numpy array with labels)
    images, labels = load_test_dataset(TEST_IMAGES_PATH)

    predicted_labels = classify_digits(images=images)

    accuracy = accuracy_score(labels, predicted_labels)

    assert np.round(accuracy * 100) >= 95


def load_test_dataset(path: str):
    # Load the test dataset from the given path
    images = []
    labels = []

    for i in range(10):  # Assuming there are 10 classes (0-9)
        class_path = f"{path}/{i}"
        
        for file_name in os.listdir(class_path):
            image = PIL.Image.open(f"{class_path}/{file_name}").convert('L').resize((28, 28))
            images.append(np.array(image))

            labels.append(i)

    return np.array(images), np.array(labels)
