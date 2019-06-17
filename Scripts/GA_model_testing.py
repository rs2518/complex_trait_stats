#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:34:07 2019

@author: raphaelsinclair
"""

import numpy as np
import pandas as pd
import os

from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model
from keras.utils import plot_model

import matplotlib.pyplot as plt

# ============= LOAD DATA =============== #
path = "~/Desktop/MSc Health Data Analytics - IC/HDA/Term 3 MSc Project/Data/GeneAtlas/J40-J47 Chronic lower respiratory diseases/clinical_c_Block_J40-J47"

data = pd.read_csv(path + '<data.file.ext>', sep = " ")


# ============= BUILD LAYERS =============== #
# Create layers
rs_id_input = Input(shape=(1,))
simple_output = Dense(1)(rs_id_input)


# Build model and plot visual image of model
sim_model = Model(rs_id_input, simple_output)
sim_model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
print(sim_model.summary())

plot_model(sim_model, to_file='sim_model.png')
image = plt.imread('sim_model.png')
plt.imshow(data)
plt.show()


# Fit and evaluate model
sim_model.fit(merged_data['SNP'], merged_data['Association?'],epochs=1, batch_size=128, validation_split=0.1, verbose=True)




