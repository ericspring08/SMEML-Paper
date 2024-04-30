from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Set the full path to the directory containing kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = '/workspaces/MTF-PAPER/KAGGLE/'

# Initialize the Kaggle API
api = KaggleApi()

# Authenticate with your Kaggle credentials
api.authenticate()

# Search for datasets by keyword
datasets = api.dataset_list(search='disease')

# Print the list of datasets
for dataset in datasets:
    print(dataset.ref)

# Download a specific dataset by its reference
#api.dataset_download_files('username/dataset-name', path='path_to_save_files')
