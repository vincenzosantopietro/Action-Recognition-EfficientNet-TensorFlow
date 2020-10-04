# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data_loader import DataLoader
import tensorflow as tf
from tensorflow.keras import *
import argparse


def main(arguments):
    # Use a breakpoint in the code line below to debug your script.
    loader = DataLoader(arguments.dataset_path, batch_size=8)
    train_dataset = loader.get_train_dataset()
    num_classes = len(loader.get_classes())

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
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

    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train_dataset, epochs=2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="resources/UCF")
    args = parser.parse_args()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/