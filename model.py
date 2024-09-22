import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D

# Function to generate batches of images for training/validation
# It resizes the images to 24x24, converts them to grayscale, and performs image augmentation
def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=32, target_size=(24, 24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

# Set batch size and target image size
BS = 32
TS = (24, 24)

# Generating training and validation batches from directories
train_batch = generator('data/train', shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# SPE (Steps Per Epoch) and VS (Validation Steps) calculate how many steps are needed to go through the entire dataset
SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS

# Define the Convolutional Neural Network (CNN) architecture
model = Sequential([
    # First convolutional layer with 32 filters and ReLU activation
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    # Max-pooling layer to reduce dimensionality
    MaxPooling2D(pool_size=(1, 1)),
    # Second convolutional layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # Third convolutional layer with 64 filters
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    # Dropout layer to prevent overfitting
    Dropout(0.25),
    # Flatten layer to convert 2D matrices to 1D vectors
    Flatten(),
    # Fully connected dense layer with 128 neurons
    Dense(128, activation='relu'),
    # Another dropout layer
    Dropout(0.5),
    # Output layer with softmax activation for 2 classes (Open/Close)
    Dense(2, activation='softmax')
])

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training and validation batches
model.fit(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Save the trained model to the models directory
model.save(os.path.join('models', 'cnnCat2.h5'), overwrite=True)
