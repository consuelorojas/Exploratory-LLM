import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from PIL import Image
import tensorflow as tf
import digits_classifier.constants as constants
from digits_classifier.interfaces import IClassifyDigits
from tests.test_data import get_test_images, get_expected_outputs


class TestDigitClassification:
    def test_classification_accuracy(self):
        model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
        
        # Load the MNIST dataset for testing (replace with your own data if needed)
        images, labels = get_test_images(), get_expected_outputs()
        
        predictions = []
        for image in images:
            img_array = np.array(image.convert('L').resize((28, 28)))
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, -1)
            
            prediction = model.predict(img_array)
            predicted_digit = int(np.argmax(prediction))
            predictions.append(predicted_digit)

        correct_predictions = sum([p == l for p, l in zip(predictions, labels)])
        
        accuracy = (correct_predictions / len(labels)) * 100
        
        assert accuracy >= 95
```

However, the above test case is tightly coupled with your model and data. A more general approach would be to use a class that implements `IClassifyDigits` interface.

```python
import numpy as np
from PIL import Image
import tensorflow as tf
from digits_classifier.interfaces import IClassifyDigits
from tests.test_data import get_test_images, get_expected_outputs


class TestDigitClassification:
    def test_classification_accuracy(self, classify_digits: IClassifyDigits):
        # Load the MNIST dataset for testing (replace with your own data if needed)
        images = np.array([np.array(image.convert('L').resize((28, 28))) for image in get_test_images()])
        
        predictions = classify_digits(images)

        labels = np.array(get_expected_outputs())
        
        correct_predictions = sum([p == l for p, l in zip(predictions, labels)])
        
        accuracy = (correct_predictions / len(labels)) * 100
        
        assert accuracy >= 95
