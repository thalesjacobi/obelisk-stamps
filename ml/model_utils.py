from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package="obelisk_stamps")
class L2Normalize(tf.keras.layers.Layer):
    """Keras-serializable L2 normalization layer (safe for Keras 3 loading)."""
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def build_embedding_model(
    image_size: Tuple[int, int] = (224, 224),
    embedding_dim: int = 256,
    base_trainable: bool = False,
) -> tf.keras.Model:
    """
    EfficientNet backbone + projection head -> L2-normalized embedding vector.
    No Lambda layers (loads in Keras 3 safe_mode=True).
    """
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image")

    # Your pipeline feeds [0..1] floats; EfficientNet expects [0..255] scale
    x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(inputs)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base.trainable = base_trainable

    x = base.output
    x = tf.keras.layers.Dense(512, activation="relu", name="proj_dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="proj_dropout")(x)
    x = tf.keras.layers.Dense(embedding_dim, name="proj_dense_2")(x)

    outputs = L2Normalize(axis=-1, name="l2_normalize")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="stamp_embedding_model")
