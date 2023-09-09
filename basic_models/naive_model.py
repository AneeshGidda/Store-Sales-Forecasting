import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_tools import evaluate
pd.options.mode.chained_assignment = None

# Load the training and testing data
x_train = pd.read_csv("x_train_basic.csv")
x_test = pd.read_csv("x_test_basic.csv")
y_train = pd.read_csv("y_train_basic.csv")
y_test = pd.read_csv("y_test_basic.csv")

# Get unique product and store values from the testing data
products = x_test["family"].unique()
stores = x_test["store_nbr"].unique()

# Create a modified testing dataset with shifted sales values
modified_x_test = pd.DataFrame()
for product in tqdm(products, desc="processing"):
    for store in stores:
        data = x_test[(x_test["store_nbr"] == store) & (x_test["family"] == product)]
        indices = data.index
        sales_data = y_test["sales"][indices]
        
        # Shift the sales data by one time step
        data["sales"] = sales_data
        data["sales"] = data["sales"].shift(1)
        modified_x_test = pd.concat([modified_x_test, data], axis=0)

# Calculate the mean sales for each group of date, store_nbr, and family
modified_x_test = modified_x_test.groupby(["date", "store_nbr", "family"])["sales"].mean().reset_index()

# Print the modified testing dataset and original x_test
print(modified_x_test)
print(x_test)

# Update x_test to use the modified testing dataset
x_test = modified_x_test.drop(columns=["sales"])

# Create a naive sales forecast based on the shifted data
naive_forecast = modified_x_test["sales"]

# Define a function to visualize the actual sales and naive forecast for a specific product and store
def visualize(x_test, y_test, naive_forecast, product, store): 
    x_test = x_test[(x_test["store_nbr"] == store) & (x_test["family"] == product)]
    indices = x_test.index
    y_test = y_test.loc[indices, "sales"]
    naive_forecast = naive_forecast.loc[indices]

    plt.figure(figsize=(12, 8))
    plt.plot(x_test.index, y_test, label="Actual Sales")
    plt.plot(x_test.index, naive_forecast, label="Naive Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(f"Actual Sales vs Naive Forecast ({product} at Store {store})")
    plt.legend()
    plt.show()

# Visualize the actual sales and naive forecast for a specific product and store
visualize(x_test, y_test, naive_forecast, "SEAFOOD", 10)

# Remove NaN values from the naive forecast and corresponding y_test values
nan_indices = naive_forecast[naive_forecast.isna()].index.tolist()
naive_forecast.dropna(inplace=True)
y_test = y_test.drop(nan_indices)

# Evaluate the performance of the naive forecast
naive_results = evaluate(y_true=y_test["sales"].to_numpy(dtype=np.float32),
                         y_pred=naive_forecast.to_numpy(dtype=np.float32))

# Print the evaluation results
print(naive_results)
