import tensorflow as tf
from nbeats_block_layer import NBeatsBlock

# Define a custom TensorFlow model for the N-BEATS architecture
class NBeatsModel(tf.keras.Model):
    def __init__(self, horizon, n_neurons, n_layers, n_stacks, input_size, theta_size):
        super(NBeatsModel, self).__init__()

        # Initialize the N-Beats block
        self.nbeats_block = NBeatsBlock(input_size=input_size,
                                        theta_size=theta_size,
                                        horizon=horizon,
                                        n_neurons=n_neurons,
                                        n_layers=n_layers)
        self.n_stacks = n_stacks

    def call(self, stack_input):
        backcast, forecast = self.nbeats_block(stack_input)  # Get the initial backcast and forecast

        # Calculate residuals by subtracting backcast from the input
        residuals = tf.keras.layers.subtract([stack_input, backcast])

        # Iterate through the N-BEATS stacks
        for _ in range(self.n_stacks - 1):
            backcast, block_forecast = self.nbeats_block(residuals)  # Calculate backcast and block forecast
            residuals = tf.keras.layers.subtract([residuals, backcast])  # Update residuals
            forecast = tf.keras.layers.add([forecast, block_forecast])  # Update forecast
        return forecast  # Return the final forecast

 
 
