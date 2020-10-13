import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from utils import get_classes
import os


class DataLoader:

    def __init__(self, base_path, image_shape=(320, 240, 3), batch_size=32):
        self.base_path = base_path
        self.training_data_path = os.path.join(base_path, "training_set")
        self.test_data_path = os.path.join(base_path, "testing_set")
        self.classes = get_classes(self.training_data_path)
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.training_data_path,
            validation_split=0.2,
            subset="training",
            seed=345,
            image_size=(self.image_shape[0], self.image_shape[1]),
            batch_size=self.batch_size,
            label_mode='categorical',
            class_names=self.classes
        )

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.training_data_path,
            validation_split=0.2,
            subset="validation",
            seed=345,
            image_size=(self.image_shape[0], self.image_shape[1]),
            batch_size=self.batch_size,
            label_mode='categorical',
            class_names=self.classes
        )

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_data_path,
            seed=123,
            image_size=(self.image_shape[0], self.image_shape[1]),
            batch_size=self.batch_size,
            label_mode='categorical',
            class_names=self.classes
        )

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        self.train_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.val_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
        self.test_ds = self.test_ds.map(lambda x, y: (normalization_layer(x), y))

        # optimizing performances
        autotune = tf.data.experimental.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=autotune)
        self.val_ds = self.train_ds.cache().prefetch(buffer_size=autotune)
        self.test_ds = self.train_ds.cache().prefetch(buffer_size=autotune)

    def get_train_dataset(self) -> tf.data.Dataset:
        return self.train_ds

    def get_val_dataset(self) -> tf.data.Dataset:
        return self.val_ds

    def get_test_dataset(self) -> tf.data.Dataset:
        return self.test_ds

    def get_classes(self) -> list:
        return self.classes
