import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam

# Define the SRCNN model
def srcnn_model():
    model = Sequential()
    # First convolutional layer with 64 filters of size 9x9
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)))
    # Second convolutional layer with 32 filters of size 5x5
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    # Third convolutional layer with 1 filter of size 5x5
    model.add(Conv2D(1, (5, 5), activation='linear', padding='same'))
    
    optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return model

# Function to load pre-trained SRCNN model
def load_pretrained_srcnn(weights_path):
    model = srcnn_model()
    model.load_weights(weights_path)
    return model

# Example usage:
# srcnn = srcnn_model()
# srcnn.summary()
# pretrained_srcnn = load_pretrained_srcnn('path_to_pretrained_weights.h5')
