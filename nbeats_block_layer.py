import tensorflow as tf

# Define a custom TensorFlow layer for an N-Beats block
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, input_size, theta_size, horizon, n_neurons, n_layers):
    super().__init__()
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Create a list of hidden Dense layers
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    
    # Create a Dense layer for generating theta values
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear")

  def call(self, inputs):
    x = inputs  # Initialize x as the input tensor
    for layer in self.hidden:
        x = layer(x)  # Apply each hidden layer with ReLU activation
    theta = self.theta_layer(x)  # Generate theta values using a linear layer
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]  # Split theta into backcast and forecast parts
    return backcast, forecast
