from numpy.typing import NDArray

import argparse
import tensorflow as tf
#import digits_classifier.interfaces as interfaces
import interfaces
import PIL.Image
import numpy as np
#import digits_classifier.constants as constants
import constants

print("Loading model from:", constants.MODEL_DIGIT_RECOGNITION_PATH)
model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
model = tf.keras.models.load_model(model_path,
                                    compile=False)

class ClassifyDigits(interfaces.IClassifyDigits):
    def __call__(self, images: NDArray) -> NDArray[np.int_]:

        images = images / 255.0                 # normalize
        images = images.reshape(-1, 28 * 28)    # flatten

        predictions = model.predict(images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)

    args = parser.parse_args()
    x = PIL.Image.open(args.image_path).convert('L').resize((28, 28))
    images = np.array(x)
    print(ClassifyDigits()(images=images))