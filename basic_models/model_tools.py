import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Function to create a model checkpoint callback
def create_model_checkpoint(model_name, save_path="saved_models"):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                                verbose=0,
                                                save_best_only=True)

# Function to create a learning rate scheduler callback
def learning_rate_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# Function to evaluate learning rate efficiency during training
def evaluate_learning_rate(history):
    learning_rates = 1e-3 * 10**(tf.range(30)/20)
    plt.semilogx(learning_rates, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Efficiency")
    plt.show()

# Function to create an early stopping callback with patience
def early_stopping(patience):
    return tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)

# Function to calculate the mean absolute scaled error (MASE)
def mean_absolute_scaled_error(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season

# Function to evaluate model performance with various metrics
def evaluate(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    if mae.ndim > 0:
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}
