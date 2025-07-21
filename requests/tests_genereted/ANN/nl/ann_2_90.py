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

def test_classification_accuracy():
    # Load the model
    model = load_model(MODEL_DIGIT_RECOGNITION_PATH)

    # Get all images from the test data directory
    image_paths = [os.path.join(TEST_DATA_PATH, f) for f in os.listdir(TEST_DATA_PATH)]

    # Initialize lists to store predictions and actual labels
    predicted_labels = []
    actual_labels = []

    # Iterate over each image path
    for i, img_path in enumerate(image_paths):
        # Extract the label from the filename (assuming it's in the format 'label_image.png')
        label = int(os.path.basename(img_path).split('_')[0])

        # Open and preprocess the image
        img = Image.open(img_path).convert('L').resize((28, 28))
        img_array = np.array(img) / 255.0

        # Make a prediction on the preprocessed image
        predictions = model.predict(np.expand_dims(img_array.reshape(-1), axis=0))

        # Get the predicted label and append it to the list of predicted labels
        predicted_label = int(np.argmax(predictions))
        predicted_labels.append(predicted_label)

        # Append the actual label to the list of actual labels
        actual_labels.append(label)

    # Calculate the accuracy using sklearn's accuracy_score function
    accuracy = accuracy_score(actual_labels, predicted_labels)

    # Assert that the accuracy is at least 95%
    assert accuracy >= 0.95

