import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define a function to plot sales data for different stores and products
def plot_sales_data(sales_data, stores, products):
    # Drop unnecessary columns from the sales_data DataFrame
    sales_data = sales_data.drop(columns=["id", "onpromotion"])
    
    # Create a large figure for the plots
    plt.figure(figsize=(26, 20))
    
    # Iterate over products and stores using tqdm for progress tracking
    for product in tqdm(products, desc="processing"):
        for store in stores:
            # Filter data for the specific store and product
            data = sales_data[(sales_data["store_nbr"] == store) & (sales_data["family"] == product)]
            
            # Plot the sales data
            plt.plot(data.index, data["sales"], label=f"store{store} {product}")
    
    # Set labels, title, and legend for the plot
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title(f"Store Sales", fontsize=18)
    plt.legend(fontsize=7)
    
    # Show the plot
    plt.show()

# Define a function to plot oil price data
def plot_oil_data(oil_data):
    # Create a plot for oil price data
    oil_data.plot(figsize=(20, 14))
    
    # Set labels, title, and legend for the plot
    plt.xlabel(oil_data.index.name)
    plt.ylabel(oil_data.columns[0])
    plt.title("Oil Prices", fontsize=18)
    plt.legend(fontsize=14)
    
    # Show the plot
    plt.show()

# Define a function to plot transaction data for different stores
def plot_transaction_data(transaction_data, stores):
    # Create a figure for the plot
    plt.figure(figsize=(20, 14))
    
    # Iterate over stores
    for store in stores:
        # Filter data for the specific store
        data = transaction_data[(transaction_data["store_nbr"] == store)]
        
        # Plot the transaction data
        plt.plot(data.index, data["transactions"], label=f"store{store}")
    
    # Set labels, title, and legend for the plot
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.title(f"Store Transactions", fontsize=18)
    plt.legend(fontsize=7)
    
    # Show the plot
    plt.show()

# Define a function to plot holiday data
def plot_holiday_data(holiday_data):
    # Preprocess the holiday data, calculating a "holiday score" based on locale and type
    holiday_data = holiday_data[holiday_data["transferred"] != True]
    holiday_data = holiday_data.drop(columns=["locale_name", "description", "transferred"])

    region_score = {"Local": 3, "Regional": 6, "National": 10}
    holiday_data["locale"] = holiday_data["locale"].replace(region_score)

    type_score = {"Work Day": 1, "Event": 2, "Bridge": 4, "Additional": 6, "Transfer": 10, "Holiday": 10}
    holiday_data["type"] = holiday_data["type"].replace(type_score)

    holiday_data["holiday_score"] = holiday_data["locale"] + holiday_data["type"]
    holiday_data = holiday_data.drop(columns=["locale", "type"])
    holiday_data = holiday_data.groupby(holiday_data.index).sum()
    
    # Create a plot for the holiday score data
    plt.figure(figsize=(20, 14))
    plt.plot(holiday_data.index, holiday_data["holiday_score"])
    
    # Set labels, title, and legend for the plot
    plt.xlabel("Date")
    plt.ylabel("Holiday Score")
    plt.title(f"Holiday Data", fontsize=18)
    
    # Show the plot
    plt.show()

# Define a function to plot store types
def plot_store_data(store_data):
    # Drop unnecessary columns from the store_data DataFrame
    store_data = store_data.drop(columns=["city", "state", "cluster"])
    
    # Define colors for store types
    colors = {'A': 'red', 'B': 'green', 'C': 'blue', 'D': 'orange', 'E': 'purple'}
    
    # Create a plot for store types
    plt.figure(figsize=(20, 14))
    
    # Iterate over rows in store_data and plot each store type
    for _, row in store_data.iterrows():
        store_nbr = row["store_nbr"]
        store_type = row["type"]
        plt.bar(store_nbr, 1, color=colors[store_type], label=store_type)

    # Set labels, title, and legend for the plot
    plt.xlabel("Store Number")
    plt.yticks([])
    plt.title("Store Types", fontsize=18)
    
    # Create a legend with custom labels based on store types and colors
    legend_labels = [plt.Line2D([0], [0], color=colors[store_type], label=store_type) for store_type in colors.keys()]
    plt.legend(handles=legend_labels)
    
    # Show the plot
    plt.show()

# Load and plot various data sets
sales_data = pd.read_csv(r"csv_files\train.csv", parse_dates=["date"], index_col=["date"])
products = sales_data["family"].unique()
stores = sales_data["store_nbr"].unique()
plot_sales_data(sales_data, stores, products)

oil_data = pd.read_csv(r"csv_files\oil.csv", parse_dates=["date"], index_col=["date"])
plot_oil_data(oil_data)

transaction_data = pd.read_csv(r"csv_files\transactions.csv", parse_dates=["date"], index_col=["date"])
stores = transaction_data["store_nbr"].unique()
plot_transaction_data(transaction_data, stores)

holiday_data = pd.read_csv(r"csv_files\holidays_events.csv", parse_dates=["date"], index_col=["date"])
plot_holiday_data(holiday_data)

store_data = pd.read_csv(r"csv_files\stores.csv")
plot_store_data(store_data)
