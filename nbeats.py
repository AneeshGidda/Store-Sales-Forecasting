import numpy as np
import tensorflow as tf
import joblib
from nbeats_model import NBeatsModel

# Specify the directory for caching
cache_directory = "cache_directory"
memory = joblib.Memory(cache_directory)

# Function to load data from cached files
@memory.cache
def load_data():
    return (tf.convert_to_tensor(np.load("o-t-h-s-x_train.npy", allow_pickle=True)), 
            tf.convert_to_tensor(np.load("o-t-h-s-x_test.npy", allow_pickle=True)),
            tf.convert_to_tensor(np.load("o-t-h-s-y_train.npy", allow_pickle=True)), 
            tf.convert_to_tensor(np.load("o-t-h-s-y_test.npy", allow_pickle=True)))

# Set a random seed for reproducibility
tf.random.set_seed(42)

# Load the training and testing data
x_train, x_test, y_train, y_test = load_data()

# Define window, horizon, and feature sizes
window = np.shape(x_test)[1]
horizon = np.shape(y_test)[1]
features = np.shape(x_test)[2]

# Define hyperparameters
N_EPOCHS = 200
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 30

INPUT_SIZE = window * horizon
THETA_SIZE = features

# Create an instance of the N-BEATS model
nbeats_model = NBeatsModel(horizon=horizon,
                           n_neurons=N_NEURONS,
                           n_layers=N_LAYERS,
                           n_stacks=N_STACKS,
                           input_size=INPUT_SIZE,
                           theta_size=THETA_SIZE)

# Compile the model with mean absolute error (MAE) loss and Adam optimizer
nbeats_model.compile(loss="mae",
                     optimizer=tf.keras.optimizers.Adam(0.001),
                     metrics=["mae", "mse"])

# Train the model with early stopping and learning rate reduction on plateau
nbeats_model.fit(x_train,
                 y_train,
                 epochs=N_EPOCHS,
                 validation_data=(x_test, y_test),
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5 , verbose=1)])
