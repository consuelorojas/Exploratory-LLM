import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("test_size", [100, 500])
def test_classification_accuracy(model, test_size):
    # Generate random MNIST-like data for testing purposes.
    np.random.seed(0)
    images = np.random.rand(test_size, 28 * 28).astype(np.float32) / 255.0
    labels = np.random.randint(low=0, high=10, size=test_size)

    predictions = model.predict(images.reshape(-1, 28, 28))
    predicted_labels = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predicted_labels)
    assert accuracy >= 0.95
```

However, the above test case may not be accurate as it uses random data for testing purposes which might not reflect real-world scenarios.

To accurately measure model performance on MNIST dataset you should use actual MNIST test set:

```python
import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score
import tensorflow_datasets as tfds


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


def test_classification_accuracy(model):
    # Load MNIST dataset.
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape the images for prediction.
    test_images = test_images.astype(np.float32) / 255.0
    predictions = model.predict(test_images.reshape(-1, 28 * 28).reshape((-1, 28, 28)))
    
    predicted_labels = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    assert accuracy >= 0.95

