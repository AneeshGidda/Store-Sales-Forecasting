import shutil

# Specify the path to the zip file
zip_file_path = r"C:\Users\Aneesh\Downloads\store-sales-time-series-forecasting.zip"

# Specify the destination directory where the contents of the zip file will be extracted
destination_directory = r"C:\Users\Aneesh\Codes\Store Sales - Time Series Forecasting"

# Use shutil.unpack_archive to extract the contents of the zip file to the destination directory
shutil.unpack_archive(zip_file_path, destination_directory)
