import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_DATA_PATH
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import pytest


@pytest.fixture
def model():
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("image_path", [os.path.join(TEST_DATA_PATH, f) for f in os.listdir(TEST_DATA_PATH)])
def test_classification_accuracy(model, image_path):
    # Load the test data and labels
    images = []
    labels = []

    for file_name in os.listdir(TEST_DATA_PATH):
        img = Image.open(os.path.join(TEST_DATA_PATH, file_name)).convert('L').resize((28, 28))
        label = int(file_name.split('.')[0])
        
        # Normalize the image data
        normalized_img = np.array(img) / 255.0
        
        images.append(normalized_img)
        labels.append(label)

    # Reshape and convert to numpy arrays for prediction
    test_images = np.reshape(images, (-1, 28 * 28))
    
    predictions = model.predict(test_images)
    predicted_labels = [int(np.argmax(prediction)) for prediction in predictions]

    accuracy = accuracy_score(labels, predicted_labels)
    assert round(accuracy, 2) >= 0.95
