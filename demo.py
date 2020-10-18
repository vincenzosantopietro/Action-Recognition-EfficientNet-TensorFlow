from data_loader import DataLoader
import tensorflow as tf
import argparse
from utils import get_callbacks_list, get_model_from_id, get_classes, preprocess_image, write_class_on_img
import numpy as np
import cv2
import os

INPUT_SHAPE = (220, 220, 3)


def main(arguments):
    cap = cv2.VideoCapture(arguments.video)

    model: tf.keras.Model = tf.keras.models.load_model(arguments.weights_path)
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    classes = get_classes(os.path.join(arguments.data_path, "training_set"))
    success, img = cap.read()

    while success:
        img_pp = preprocess_image(img, INPUT_SHAPE)
        # Inference
        x = model.predict(np.expand_dims(img_pp, 0), batch_size=1)
        # Post-process image
        img_out = write_class_on_img(img_pp, classes[int(np.argmax(np.array(x)))])
        cv2.imshow("EfficientNet Prediction", img_out)
        cv2.waitKey(10)
        # Read next frame
        success, img = cap.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="resources/basket.mp4")
    parser.add_argument("--data_path", type=str, default="resources/UCF")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--efficientnet_id", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Id of the desired EfficientNetB<id> model", default=0)
    args = parser.parse_args()
    main(args)
