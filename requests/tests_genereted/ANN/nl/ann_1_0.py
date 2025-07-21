import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
from PIL import Image
import pytest
import os


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
    correct_classifications = 0
    
    # Load all the images from a directory
    image_paths = [os.path.join(TEST_IMAGES_PATH, file) 
                   for file in os.listdir(TEST_IMAGES_PATH)]
    
    labels = []
    predictions = classify_digits(np.array([np.array(Image.open(image_path).convert('L').resize((28, 28))) 
                                           for image_path in image_paths]))
    
    # Assuming the images are named as 'label_image_name.png'
    for path in image_paths:
        label = int(path.split('/')[-1].split('_')[0])
        labels.append(label)
        
    correct_classifications = sum([p == l for p, l in zip(predictions, labels)])
    
    accuracy = (correct_classifications / len(labels)) * 100
    
    assert accuracy >= 95
