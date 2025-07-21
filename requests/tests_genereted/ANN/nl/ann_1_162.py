import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
from PIL import Image
import pytest
import os


class TestModelAccuracy:
    @pytest.fixture
    def classify_digits(self):
        model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
        
        class ClassifyDigits(IClassifyDigits):
            def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
                images = images / 255.0                 # normalize
                images = images.reshape(-1, 28 * 28)    # flatten

                predictions = model.predict(images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])
        
        yield ClassifyDigits()

    def test_model_accuracy(self, classify_digits):
        correct_classifications = 0
        total_images = 0
        
        for filename in os.listdir(TEST_IMAGES_PATH):
            if filename.endswith(".png"):
                image_path = os.path.join(TEST_IMAGES_PATH, filename)
                label = int(filename.split("_")[0])
                
                image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))
                prediction = classify_digits(np.array([image_array]))
                
                if prediction[0] == label:
                    correct_classifications += 1
                
                total_images += 1
        
        accuracy = (correct_classifications / total_images) * 100
        assert accuracy >= 95.0

