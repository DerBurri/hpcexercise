# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:10:40 2022

@author: sshakibhamedan
"""

import time
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from scipy.special import softmax


# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_,_), (test_images, test_labels) = mnist.load_data()

# Load pre-trained model
model = tf.keras.models.load_model('my_org_model_top4_quant.h5')

# Get model weights and print model summary
model_weights = model.get_weights()
model.summary()

# use 1000 test image from mnist dataset
batch_size = 1000
input_features = test_images[:batch_size]

# Load pre-computed multiplier arrays
multipliers = [np.load(f"S_{i}.npy") for i in range(41, 45)] + \
              [np.load(f"S_{i}.npy") for i in range(51, 55)] + \
              [np.load(f"S_{i}.npy") for i in range(61, 65)] + \
              [np.load(f"S_{i}.npy") for i in range(71, 75)] + \
              [np.load(f"S_{i}.npy") for i in range(81, 85)]

def custom_elementwise_multiplication(a, b, t=1):
    """
    Custom element-wise multiplication function.
    Multiplies elements of a and b using pre-computed arrays based on type t.
    """
    a = np.array(a)
    b = np.array(b)
    a_shape = np.shape(a)
    b = np.reshape(b, a_shape)
    result = np.zeros(a_shape)
    
    # Select the appropriate multiplier array based on t
    multiplier = multipliers[t - 1]
    
    # Perform element-wise multiplication using the selected multiplier
    if len(a_shape) == 1:
        for i in range(a_shape[0]):
            result[i] = multiplier[int(a[i]) + 128, int(b[i]) + 128]
    if len(a_shape) == 2:
        for i in range(a_shape[0]):
            for j in range(a_shape[1]):
                result[i, j] = multiplier[int(a[i, j]) + 128, int(b[i, j]) + 128]
    return result

def custom_matrix_multiplication(a, b, t=1):
    """
    Custom matrix multiplication using custom_elementwise_multiplication function.
    """
    a = np.array(a)
    b = np.array(b)
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    result = np.zeros([a_shape[0], b_shape[1]])
    
    # Perform matrix multiplication
    for i in range(a_shape[0]):
        for j in range(b_shape[1]):
            result[i, j] = np.sum(custom_elementwise_multiplication(a[i, :], b[:, j], t))
    return result

def custom_conv2d(a, b, t=1):
    """
    Custom 2D convolution using custom_elementwise_multiplication function.
    """
    a = np.array(a)
    b = np.array(b)
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    result_shape1 = np.abs(a_shape[0] - b_shape[0]) + 1
    result_shape2 = np.abs(a_shape[1] - b_shape[1]) + 1
    result = np.zeros([result_shape1, result_shape2])
    
    # Perform 2D convolution
    for i in range(result_shape1):
        for j in range(result_shape2):
            result[i, j] = np.sum(custom_elementwise_multiplication(np.flip(b), a[i:i + b_shape[0], j:j + b_shape[1]], t))
    return result

def model_forward_pass(image_index, t=1):
    """
    Custom function to perform forward pass through the modified model.
    """
    # Perform a series of convolutions and activations
    conv1_input = np.floor(input_features[image_index] / 2)
    conv1_input = conv1_input.reshape(28, 28, 1)
    conv1_output = np.zeros([28, 28, 64])
    for i in range(64):
        for j in range(1):
            conv1_output[:, :, i] += custom_conv2d(np.array(conv1_input[:, :, j]), np.flip(model_weights[0][:, :, j, i]), t)
        conv1_output[:, :, i] += model_weights[1][i]
    relu1_output = np.maximum(0, conv1_output)
    relu1_output = np.round((relu1_output / np.max(relu1_output)) * 127)
    
    conv2_output = np.zeros([28, 28, 32])
    for i in range(32):
        for j in range(64):
            conv2_output[:, :, i] += custom_conv2d(np.array(relu1_output[:, :, j]), np.flip(model_weights[2][:, :, j, i]), t)
        conv2_output[:, :, i] += model_weights[3][i]
    relu2_output = np.maximum(0, conv2_output)
    relu2_output = np.round((relu2_output / np.max(relu2_output)) * 127)
    
    conv3_output = np.zeros([28, 28, 16])
    for i in range(16):
        for j in range(32):
            conv3_output[:, :, i] += custom_conv2d(np.array(relu2_output[:, :, j]), np.flip(model_weights[4][:, :, j, i]), t)
        conv3_output[:, :, i] += model_weights[5][i]
    relu3_output = np.maximum(0, conv3_output)
    relu3_output = np.round((relu3_output / np.max(relu3_output)) * 127)
    
    conv4_output = np.zeros([26, 26, 8])
    for i in range(8):
        for j in range(16):
            conv4_output[:, :, i] += custom_conv2d(np.array(relu3_output[:, :, j]), np.flip(model_weights[6][:, :, j, i]), t)
        conv4_output[:, :, i] += model_weights[7][i]
    relu4_output = np.maximum(0, conv4_output)
    relu4_output = np.round((relu4_output / np.max(relu4_output)) * 127)
    
    conv5_output = np.zeros([24, 24, 4])
    for i in range(4):
        for j in range(8):
            conv5_output[:, :, i] += custom_conv2d(np.array(relu4_output[:, :, j]), np.flip(model_weights[8][:, :, j, i]), t)
        conv5_output[:, :, i] += model_weights[9][i]
    relu5_output = np.maximum(0, conv5_output)
    relu5_output = np.round((relu5_output / np.max(relu5_output)) * 127)
    
    flatten_output = np.reshape(relu5_output, [1, 2304])
    fc1_output = custom_matrix_multiplication(flatten_output, model_weights[10], t) + model_weights[11]
    relu6_output = np.maximum(0, fc1_output) + 0.000001
    relu6_output = np.round((relu6_output / np.max(relu6_output)) * 127)
    
    fc2_output = custom_matrix_multiplication(relu6_output, model_weights[12], t) + model_weights[13]
    relu7_output = np.maximum(0, fc2_output) + 0.000001
    relu7_output = np.round((relu7_output / np.max(relu7_output)) * 127)
    
    fc3_output = custom_matrix_multiplication(relu7_output, model_weights[14], t) + model_weights[15] + 0.000001
    fc3_output = np.round((fc3_output / np.max(fc3_output)) * 127)
    
    return np.argmax(fc3_output)


# Loop through different multiplier types and calculate results

for multiplier_type in range(1, 21):
    # Record the start time
    start_time = time.time()

    results = []
    for image_index in range(1000):
        results.append(model_forward_pass(image_index, multiplier_type))
        print(results)
        filename = f"Result_Approx_Multi_{multiplier_type}.npy"
        np.save(filename, results)
        print(results)
        print(np.size(results))

    elapsed_time = time.time() - start_time
    print('Execution time:', elapsed_time, 'seconds')
    
    # Calculate and print accuracy
    accuracy = np.sum(results == test_labels[:len(results)]) / len(results)
    print(f"Accuracy with multiplier {multiplier_type}: {accuracy}")


# Stop