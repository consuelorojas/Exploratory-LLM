#import digits_classifier.constants as constants
import constants
import tensorflow as tf

def train_digit_recognition_model(x_train, y_train, x_test, y_test) -> tf.keras.models.Model:
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape images to 1D vectors
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    model = train_digit_recognition_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    model.save(constants.MODEL_DIGIT_RECOGNITION_PATH)