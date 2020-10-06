# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data_loader import DataLoader
import tensorflow as tf
from tensorflow.keras import *
import argparse
from utils import get_callbacks_list, get_model_from_id

INPUT_SHAPE = (224, 224, 3)

def main(arguments):

    loader = DataLoader(arguments.dataset_path, image_shape=INPUT_SHAPE, batch_size=arguments.batch_size)
    train_dataset = loader.get_train_dataset()
    val_dataset = loader.get_val_dataset()
    test_dataset = loader.get_test_dataset()
    num_classes = len(loader.get_classes())

    """
    simple model for fast experimentation
    model = tf.keras.Sequential([
        # layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    """
    # Get the right EfficientNet model & compile
    model = get_model_from_id(input_shape=INPUT_SHAPE, model_id=arguments.efficientnet_id, num_classes=num_classes)
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Get the list of callbacks and fit the model
    cbks = get_callbacks_list(arguments.efficientnet_id)
    model.fit(train_dataset, validation_data=val_dataset, epochs=2, callbacks=cbks, use_multiprocessing=True)

    # Model evaluation
    results = model.evaluate(test_dataset)
    print("Accuracy on test set: {}".format(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="resources/UCF")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--efficientnet_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Id of the desired EfficientNetB<id> model", default=1)
    args = parser.parse_args()
    main(args)
