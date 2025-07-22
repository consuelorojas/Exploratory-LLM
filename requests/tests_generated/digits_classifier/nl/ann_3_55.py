import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from PIL import Image
import tensorflow as tf
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_DATA_DIR
import pytest
import os


class TestDigitClassification:
    @pytest.fixture
    def classify_digits(self):
        class ClassifyDigits(IClassifyDigits):
            def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
                model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
                images = images / 255.0                 # normalize
                images = images.reshape(-1, 28 * 28)    # flatten

                predictions = model.predict(images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])

        yield ClassifyDigits()

    def test_classification_accuracy(self, classify_digits):
        correct_classifications = 0
        total_images = 0

        for filename in os.listdir(TEST_DATA_DIR):
            if filename.endswith(".png"):
                image_path = os.path.join(TEST_DATA_DIR, filename)
                expected_digit = int(filename.split("_")[1].split('.')[0])

                image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))
                predicted_digits = classify_digits(np.expand_dims(image_array, axis=0))

                if predicted_digits[0] == expected_digit:
                    correct_classifications += 1

                total_images += 1

        accuracy = (correct_classifications / total_images) * 100
        assert accuracy >= 95.0
