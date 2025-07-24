import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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

However, the above test case may fail because it uses random data which might not be representative of real-world MNIST-like images.

To improve this test case and make sure that we are actually testing our model's performance on a dataset similar to what it was trained on (MNIST), you should use actual MNIST test data. Here is an improved version:

```python
import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("test_size", [100, 500])
def test_classification_accuracy(model):
    # Load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    predictions = model.predict(x_test)
    
    predicted_labels = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(y_test[:len(predicted_labels)], predicted_labels)
    assert accuracy >= 0.95

def test_classification_accuracy_full_dataset(model):
    # Load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    predictions = model.predict(x_test)
    
    predicted_labels = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(y_test[:len(predicted_labels)], predicted_labels)
    assert accuracy >= 0.95

