import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')





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
    total_images = 0
    
    for filename in os.listdir(TEST_IMAGES_PATH):
        if filename.endswith(".png"):
            image_path = os.path.join(TEST_IMAGES_PATH, filename)
            label = int(filename.split("_")[0])
            
            image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))
            prediction = classify_digits(np.expand_dims(image_array, axis=0))[0]
            
            if prediction == label:
                correct_classifications += 1
            
            total_images += 1
    
    accuracy = (correct_classifications / total_images) * 100
    assert accuracy >= 95.0


def test_classification_accuracy_with_batch(classify_digits):
    image_arrays = []
    labels = []
    
    for filename in os.listdir(TEST_IMAGES_PATH):
        if filename.endswith(".png"):
            label = int(filename.split("_")[0])
            
            image_path = os.path.join(TEST_IMAGES_PATH, filename)
            image_array = np.array(Image.open(image_path).convert('L').resize((28, 28)))
            
            labels.append(label)
            image_arrays.append(image_array)
    
    predictions = classify_digits(np.stack(image_arrays))
    correct_classifications = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    
    accuracy = (correct_classifications / len(labels)) * 100
    assert accuracy >= 95.0

