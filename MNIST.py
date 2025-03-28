import tensorflow as tf  # Importing TensorFlow for deep learning
from tensorflow.keras.datasets import mnist  # Importing the MNIST dataset
from tensorflow.keras.models import Sequential  # Sequential model for building our neural network
from tensorflow.keras.layers import Dense, Flatten  # Layers for the neural network
from sklearn.metrics import confusion_matrix  # For generating the confusion matrix
import numpy as np  # Numerical operations

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Splitting the dataset into training and testing sets

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0  # Scaling the pixel values to [0, 1]

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flattening 28x28 images into 1D vectors
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 neurons for digit classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Using Adam optimizer

# Train the model
model.fit(x_train, y_train, epochs=5)  # Training the model for 5 epochs

# Evaluate the model
y_pred = np.argmax(model.predict(x_test), axis=1)  # Predicting classes for test images

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)  # Generating the confusion matrix
print("Confusion Matrix:\n", conf_matrix)  # Displaying the confusion matrix
