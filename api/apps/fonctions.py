#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Mar 24 16:00:37 2021

@author: bonfils
"""

import tensorflow as tf
from tensorflow import keras

## Generator model
def load_generator_model(model_path):
    model = keras.models.load_model(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),  # Low learning rate
        loss= keras.metrics.Mean(name="g_loss"),
    )
    return model


# fig = plt.figure(figsize=[16, 8])
# for i in range(batch_size):
#     ax = plt.subplot(4, 8, i+1)
#     plt.imshow((generated_images * 255).astype(np.uint8)[i])
#     plt.axis("off")