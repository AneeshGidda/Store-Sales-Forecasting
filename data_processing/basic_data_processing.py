import pandas as pd

# Load the sales data from a CSV file, parse the 'date' column as dates, and set it as the index
# sales_data: date - id - store_nbr - family - sales - onpromotion
data = pd.read_csv(r"csv_files\train.csv", parse_dates=["date"], index_col=["date"])

# Drop the 'id' and 'onpromotion' columns from the DataFrame
data = data.drop(columns=["id", "onpromotion"])

# Define a function to format the data for training and testing
def basic_data_format(data, split_percentage=0.8):
    # Calculate the split size based on the specified percentage
    split_size = int(split_percentage * len(data))
    
    # Ensure that the split doesn't occur in the middle of a day
    while split_size < len(data) - 1:
        if data.index[split_size] == data.index[split_size + 1]:
            split_size += 1
        else:
            break
    
    # Group the data by 'date', 'store_nbr', and 'family', calculating the mean of 'sales'
    data = data.groupby(["date", "store_nbr", "family"])["sales"].mean().reset_index()
    
    # Separate the features (x_data) and the target (y_data)
    x_data = data.drop(columns=["sales"])
    y_data = data["sales"]

    # Split the data into training and testing sets
    x_train, y_train = x_data[:split_size], y_data[:split_size]
    x_test, y_test = x_data[split_size:], y_data[split_size:]
    return x_train, y_train, x_test, y_test

# Call the function to get the formatted training and testing data
x_train, y_train, x_test, y_test = basic_data_format(data)

# Save the formatted data to CSV files
x_train.to_csv("x_train_basic.csv")
y_train.to_csv("y_train_basic.csv", index=False)
x_test.to_csv("x_test_basic.csv")
y_test.to_csv("y_test_basic.csv", index=False)
