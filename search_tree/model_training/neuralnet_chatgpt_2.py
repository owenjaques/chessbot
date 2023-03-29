import numpy as np
import pandas as pd
import chess
import chess.engine
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

def model_cnn2(data, labels, valid, model_name):
        # Define the input shape
        # 8x8 chess board with 8 channels (6 pieces + 2 for current player and opponent)
        input_shape = (8, 8, 12)

        # Define the model
        model = Sequential()

        # Convolutional layers
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(42, activation='softmax'))

        # Output layer
        model.add(Dense(42, activation='softmax')) 
                                                #categorical_crossentropy
        # Compile the model                        #binary_crossentropy
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])


        # Train the model
        history = model.fit(data, labels, epochs=10, batch_size=32, validation_data=valid)

        model.save(model_name)

        return history, model
