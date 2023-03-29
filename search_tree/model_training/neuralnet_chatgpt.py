import numpy as np
import pandas as pd
import chess
import chess.engine
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

def model_cnn1(data, labels, model_name):
    # Define the input shape
    # 8x8 chess board with 8 channels (6 pieces + 2 for current player and opponent)
    input_shape = (8, 8, 8)

    # Define the model
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu',
            padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))

    # Output layer
    model.add(Dense(6, activation='softmax'))  # 6 possible moves

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])


    # Train the model
    history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.1)

    model.save(model_name)

    return history, model
