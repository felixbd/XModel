#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.datasets import fetch_openml
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau


class XModel(tf.keras.Model):
    def __init__(self, base_model):
        super(XModel, self).__init__()
        self.base_model = base_model

    def compile(self, optimizer, loss_fn):
        # super(XModel, self).compile()

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(f"Invalid optimizer type: {type(optimizer).__name__}. "
                            f"Please provide a valid TensorFlow optimizer instance.")

        super(XModel, self).compile(optimizer=optimizer, loss="mae")
        # self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, inputs, training=False):
        # Delegate the forward pass to the base model
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            # Pass x, y_true, and y_pred to the loss function
            loss = self.loss_fn(x, y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
