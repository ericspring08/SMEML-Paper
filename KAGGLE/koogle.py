from kaggle.api.kaggle_api_extended import KaggleApi
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/workspaces/MTF-PAPER/KAGGLE/'

# Initialize the Kaggle API
api = KaggleApi()

# Search for datasets by keyword
datasets = api.dataset_list(search='disease')

# Print the list of datasets
for dataset in datasets:
    print(dataset.ref)

# Download a specific dataset by its reference
#api.dataset_download_files('username/dataset-name', path='path_to_save_files')
