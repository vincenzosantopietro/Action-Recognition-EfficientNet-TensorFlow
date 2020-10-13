from data_loader import DataLoader
import tensorflow as tf
import argparse
from utils import get_callbacks_list, get_model_from_id
import tensorflow.keras.layers as layers

INPUT_SHAPE = (220, 220, 3)


def main(arguments):
    loader = DataLoader(arguments.dataset_path, image_shape=INPUT_SHAPE, batch_size=arguments.batch_size)
    train_dataset = loader.get_train_dataset()
    val_dataset = loader.get_val_dataset()
    test_dataset = loader.get_test_dataset()
    num_classes = len(loader.get_classes())

    # simple model for fast experimentation
    """
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    """

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    # Get the right EfficientNet model & compile
    base_model = get_model_from_id(input_shape=INPUT_SHAPE, include_top=False, model_id=arguments.efficientnet_id,
                                   num_classes=num_classes, final_activation='softmax')
    out = base_model(inputs)
    out = layers.AveragePooling2D()(out)
    out = layers.Flatten()(out)
    x = layers.Dropout(0.5)(out)
    out = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1_l2())(x)

    model = tf.keras.Model(inputs, out)
    model.summary()
    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Get the list of callbacks and fit the model
    cbks = get_callbacks_list(arguments.efficientnet_id)
    try:
        model.fit(train_dataset, validation_data=val_dataset, epochs=arguments.epochs, callbacks=cbks)

        # Model evaluation
        results = model.evaluate(test_dataset)
        print("Accuracy on test set: {}".format(results[-1]))
    except Exception as e:
        print("{}".format(e))
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="resources/UCF")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--efficientnet_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Id of the desired EfficientNetB<id> model", default=0)
    args = parser.parse_args()
    main(args)
