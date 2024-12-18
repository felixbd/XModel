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
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(f"Invalid optimizer type: {type(optimizer).__name__}. "
                            f"Please provide a valid TensorFlow optimizer instance.")
        super(XModel, self).compile(optimizer=optimizer)  # Use parent's compile
        self.loss_fn = loss_fn  # Custom loss function

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        # Unpack data
        x, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.base_model(x, training=True)
            loss = self.loss_fn(x, y_true, y_pred)  # Custom loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Log metrics
        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss
        return metrics

    def test_step(self, data):
        # Validation step using custom loss
        x, y_true = data
        y_pred = self.base_model(x, training=False)
        loss = self.loss_fn(x, y_true, y_pred)  # Custom loss

        # Log metrics
        self.compiled_metrics.update_state(y_true, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss
        return metrics

    def predict_with_inputs(self, x):
        # Custom method for predictions with access to x
        y_pred = self.base_model(x, training=False)
        return y_pred, x
