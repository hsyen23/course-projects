# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 23:50:02 2022

@author: Yen
"""

from model_setup import ModelSetup
from os.path import exists
import numpy as np

model_setting = ModelSetup()



# This section for first time, it will create "model_parameters_w.txt" and "model_parameters_b.txt"
#uncommon it to run
model_setting.create_parameters(3,8)


# load previous trained parameters
w, b = model_setting.load_parameters()

# get training data and convert it into suitable structure for my code
data, label = model_setting.get_trainingData()
num_data = len(data)

# shuffle data
data, label = model_setting.shuffle_data(data, label)
    
    
# train my model and update parameters (weight and bias)
# you can customize batch size, learning rate and number of iteration
batch_size = 200
lr = 1
num_iter = 1000
    
# training part, you can only repeat this part to train more time
i = 0
while i < num_data:
    if i + batch_size < num_data:
        w, b = model_setting.train_model(w, b, [data[i:i+batch_size], label[i:i+batch_size]], lr, num_iter)
        i += batch_size
    else:
        # final case which can't split data into full batch size
        w, b = model_setting.train_model(w, b, [data[i:], label[i:]], lr, num_iter)
        i += batch_size

# save updated parameters into file
model_setting.save_parameters(w, b)