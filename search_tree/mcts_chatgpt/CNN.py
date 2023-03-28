import tensorflow as tf
import numpy as np
import chess.pgn
import os
import random

# Load chess board images and their corresponding moves
# Images and moves should be preprocessed and stored in separate directories
# In this example, we assume they are stored in 'images/' and 'moves/' respectively
images_dir = 'images/'
moves_dir = 'moves/'

image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
move_paths = [os.path.join(moves_dir, f) for f in os.listdir(moves_dir)]

images = np.array([tf.keras.preprocessing.image.load_img(
    path, target_size=(224, 224)) for path in image_paths])
moves = np.array([np.load(path) for path in move_paths])

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

# Train the model
model.fit(images, moves, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('model_chatgpt.h5')


