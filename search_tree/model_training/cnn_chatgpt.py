import tensorflow as tf
import numpy as np
import chess.pgn
import os
import random

# Load chess board images and their corresponding moves
# Images and moves should be preprocessed and stored in separate directories
# In this example, we assume they are stored in 'images/' and 'moves/' respectively

def model_cnn1():
    # Create a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])

    return model
# Train the model
#model.fit(images, moves, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
#model.save('model_chatgpt.h5')


