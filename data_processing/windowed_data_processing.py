import numpy as np
import pandas as pd

# Load various data sets from CSV files
sales_data = pd.read_csv(r"csv_files\train.csv", parse_dates=["date"], index_col=["date"])
oil_data = pd.read_csv(r"csv_files\oil.csv", parse_dates=["date"], index_col=["date"])
transaction_data = pd.read_csv(r"csv_files\transactions.csv", parse_dates=["date"], index_col=["date"])
holiday_data = pd.read_csv(r"csv_files\holidays_events.csv", parse_dates=["date"], index_col=["date"])
store_data = pd.read_csv(r"csv_files\stores.csv", index_col=["store_nbr"])

# sales_data: date - id - store_nbr - family - sales - onpromotion
# oil_data: date - dcoilwtico
# transaction_data: date - store_nbr - transactions
# holiday_data: date - type - locale - locale_name - description - transferred
# store_data: store_nbr - city - state - type - cluster

# Define a function to get formatted data with optional additional features
def get_data(sales_data=sales_data, oil_data=oil_data, store_data=store_data, transaction_data=transaction_data, holiday_data=holiday_data, 
             include_oil_prices=False, include_store_info=False, include_transaction_info=False, include_holiday_info=False):
    # Drop the 'id' column from the sales_data DataFrame
    sales_data = sales_data.drop(columns=["id"])
    
    # One-hot encode the 'family' column in the sales_data DataFrame
    data = pd.get_dummies(sales_data, columns=["family"], prefix="", prefix_sep="")
    
    # Initialize an empty file prefix string
    file_prefix = ""

    # Include oil prices data if specified
    if include_oil_prices == True:
        oil_data.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
        data = pd.merge(data, oil_data, on="date", how="left")
        file_prefix += "o-"

    # Include transaction data if specified
    if include_transaction_info == True:
        data = pd.merge(data, transaction_data, on=["date", "store_nbr"], how="left")
        file_prefix += "t-"

    # Include holiday data if specified
    if include_holiday_info == True:
        # Preprocess the holiday data to calculate a "holiday score"
        holiday_data = holiday_data[holiday_data["transferred"] != True]
        holiday_data = holiday_data.drop(columns=["locale_name", "description", "transferred"])

        region_score = {"Local": 3, "Regional": 6, "National": 10}
        holiday_data["locale"] = holiday_data["locale"].replace(region_score)

        type_score = {"Work Day": 1, "Event": 2, "Bridge": 4, "Additional": 6, "Transfer": 10, "Holiday": 10}
        holiday_data["type"] = holiday_data["type"].replace(type_score)

        holiday_data["holiday_score"] = holiday_data["locale"] + holiday_data["type"]
        holiday_data = holiday_data.drop(columns=["locale", "type"])
        holiday_data = holiday_data.groupby(holiday_data.index).sum()
        data = pd.merge(data, holiday_data, on=["date"], how="left")
        file_prefix += "h-"

    # Include store information if specified
    if include_store_info == True:
        store_data = store_data.drop(columns=["city", "state", "cluster"])
        store_data = pd.get_dummies(store_data, columns=["type"], prefix="store_type", prefix_sep="")
        data = data.reset_index()
        data = pd.merge(data, store_data, on=["store_nbr"], how="left")
        data.set_index("date", inplace=True)
        data = data.drop(columns=["store_nbr"])
        file_prefix += "s-"
    
    return file_prefix, data

# Define a function to format data into a windowed format for time series forecasting
def windowed_data_format(data, horizon=1, window_size=7, split_percentage=0.8):
    # Calculate the split size based on the specified percentage
    split_size = int(split_percentage * len(data))
    
    # Ensure that the split doesn't occur in the middle of a day
    while split_size < len(data) - 1:
        if data.index[split_size] == data.index[split_size + 1]:
            split_size += 1
        else:
            break
    
    # Handle missing values and perform data preprocessing
    if "holiday_score" in data.columns:
        data["holiday_score"].fillna(0, inplace=True)

    if "oil_price" in data.columns:
        data["oil_price"].interpolate(method="time", inplace=True)
        data["oil_price"].fillna(method="bfill", inplace=True)

    if "transactions" in data.columns:
        data["transactions"].interpolate(method="time", inplace=True)
        data["transactions"].fillna(method="bfill", inplace=True)

    # Get the index of the 'sales' column in the DataFrame
    sales_column_index = data.columns.get_loc("sales")
    
    # Extract year, month, and day from the index
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    
    # Convert the DataFrame to a NumPy array and ensure it is of float32 data type
    data = data.astype("float32")
    data = data.to_numpy()

    # Create windowed data
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(data)-(window_size+horizon-1)), axis=0).T
    windowed_array = data[window_indexes]
    x_data, y_data = windowed_array[:, :-horizon], windowed_array[:, -horizon:]
    y_data = y_data[:, :, sales_column_index, np.newaxis]

    # Split data into training and testing sets
    x_train, y_train = x_data[:split_size], y_data[:split_size]
    x_test, y_test = x_data[split_size:], y_data[split_size:]
    
    return x_train, y_train, x_test, y_test

# Get data with specified features
file_prefix, data = get_data(include_oil_prices=True,
                             include_transaction_info=True,
                             include_holiday_info=True,
                             include_store_info=True)

# Format the data into a windowed format for time series forecasting
x_train, y_train, x_test, y_test = windowed_data_format(data)

# Save the formatted data as NumPy arrays
np.save(f"{file_prefix}x_train", x_train)
np.save(f"{file_prefix}y_train", y_train)
np.save(f"{file_prefix}x_test", x_test)
np.save(f"{file_prefix}y_test", y_test)
