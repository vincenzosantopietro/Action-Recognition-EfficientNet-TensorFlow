import tensorflow as tf
import os
from datetime import datetime
import glob
import sys

EFFICIENTNET_IDS = [0, 1, 2, 3, 4, 5, 6, 7]


def get_callbacks_list(model_id: int) -> list:
    if model_id not in EFFICIENTNET_IDS:
        raise ValueError("Wrong model ID")
    
    model_abs_path = os.path.join(os.getcwd(), "models_efficientnetb{}-{}".format(model_id, datetime.now().strftime(
        "%Y%m%d-%H%M%S")))
    log_path = os.path.join(os.getcwd(), "logs",
                            "efficientnetb{}-{}".format(model_id, datetime.now().strftime("%Y%m%d-%H%M%S")))

    if not os.path.exists(model_abs_path):
        os.mkdir(model_abs_path)

    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_abs_path, save_best_only=True, monitor='val_loss')
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_path)

    return [ckpt, tb]


def get_model_from_id(input_shape: tuple, include_top: bool, model_id: int, num_classes: int = 101,
                      final_activation: any = 'softmax') -> tf.keras.Model:
    if model_id not in EFFICIENTNET_IDS:
        raise ValueError("Wrong model ID")
    if model_id == 0:
        return tf.keras.applications.EfficientNetB0(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 1:
        return tf.keras.applications.EfficientNetB1(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 2:
        return tf.keras.applications.EfficientNetB2(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 3:
        return tf.keras.applications.EfficientNetB3(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 4:
        return tf.keras.applications.EfficientNetB4(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 5:
        return tf.keras.applications.EfficientNetB5(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 6:
        return tf.keras.applications.EfficientNetB6(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)
    elif model_id == 7:
        return tf.keras.applications.EfficientNetB7(include_top=include_top, weights=None, input_shape=input_shape,
                                                    classes=num_classes, classifier_activation=final_activation)


def get_classes(path: str) -> list:
    is_windows = sys.platform.startswith('win')
    if is_windows:
        return [c.split('\\')[-1] for c in glob.glob("{}\\*".format(path))]
    else:
        return [c.split('/')[-1] for c in glob.glob("{}/*".format(path))]
