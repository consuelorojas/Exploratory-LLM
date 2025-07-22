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
            model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

            def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
                images = images / 255.0                 # normalize
                images = images.reshape(-1, 28 * 28)    # flatten

                predictions = self.model.predict(images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])

        yield ClassifyDigits()

    def test_classification_accuracy(self, classify_digits):
        total_correct = 0
        total_images = 0

        for filename in os.listdir(TEST_DATA_DIR):
            if not filename.endswith(".png"):
                continue

            label = int(filename.split("_")[0])
            image_path = os.path.join(TEST_DATA_DIR, filename)
            image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))

            prediction = classify_digits(np.expand_dims(image_array, axis=0))[0]

            total_images += 1
            if label == prediction:
                total_correct += 1

        accuracy = (total_correct / total_images) * 100
        assert accuracy >= 95.0
