import os
from kaggle.api.kaggle_api_extended import KaggleApi


os.environ['KAGGLE_CONFIG_DIR'] = '/workspaces/MTF-PAPER/KAGGLE/'
api = KaggleApi()


api.authenticate()


datasets = api.dataset_list(search='disease')
datasets_folder = 'DATASETS'
os.makedirs(datasets_folder, exist_ok=True)

for dataset in datasets:
    dataset_name = dataset.ref.split('/')[-1]  
    download_path = os.path.join(datasets_folder, dataset_name)  

    if not os.path.exists(download_path):
        api.dataset_download_files(dataset.ref, path=download_path, quiet=False, unzip=True)
        
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith('.csv'):
                    src = os.path.join(root, file)
                    dst = os.path.join(datasets_folder, file)
                    os.replace(src, dst)

        for root, dirs, files in os.walk(download_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

print("All CSV files downloaded and saved to DATASETS folder.")
