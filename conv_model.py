import numpy as np
import tensorflow as tf
import joblib

# Define a cache directory for joblib to store cached results
cache_directory = "cache_directory"
memory = joblib.Memory(cache_directory)

# Define a function to load and cache data using joblib
@memory.cache
def load_data():
    return (tf.convert_to_tensor(np.load("o-t-h-s-x_train.npy", allow_pickle=True)), 
            tf.convert_to_tensor(np.load("o-t-h-s-x_test.npy", allow_pickle=True)),
            tf.convert_to_tensor(np.load("o-t-h-s-y_train.npy", allow_pickle=True)), 
            tf.convert_to_tensor(np.load("o-t-h-s-y_test.npy", allow_pickle=True)))

# Set a random seed for TensorFlow for reproducibility
tf.random.set_seed(42)

# Load data using the defined function
x_train, x_test, y_train, y_test = load_data()

# Get the number of time steps in the forecasting horizon
horizon = np.shape(y_test)[1]

# Define a sequential convolutional neural network (CNN) model
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation="relu"),  # Convolutional layer 1
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"),   # Convolutional layer 2
    tf.keras.layers.Dense(horizon)  # Output layer with 'horizon' units
])

# Compile the model with Mean Absolute Error (MAE) loss and the Adam optimizer
conv_model.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# Train the model on the training data for 10 epochs
conv_model.fit(x_train, y_train, epochs=10)