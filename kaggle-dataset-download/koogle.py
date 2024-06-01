import os
from kaggle.api.kaggle_api_extended import KaggleApi
from uuid import uuid4

api = KaggleApi()

api.authenticate()

datasets = []
page_count = 1
for i in range(1, page_count+1):
    dataset_page = api.dataset_list(
        search='disease', file_type='csv', max_size=1000000)
    datasets.extend(dataset_page)

datasets_folder = 'DATASETS'
os.makedirs(datasets_folder, exist_ok=True)

for index, dataset in enumerate(datasets):
    dataset_name = dataset.ref.split('/')[-1]
    download_path = os.path.join(datasets_folder, dataset_name)

    if not os.path.exists(download_path):
        api.dataset_download_files(
            dataset.ref, path=download_path, quiet=True, unzip=True)
        print(f"[{index}] Downloaded {dataset_name}")

        # go through the files in the dataset folder
        for file in os.listdir(download_path):
            if file.endswith('.csv'):
                # random genereate uuid for the file
                uuid_name = str(uuid4())
                os.rename(os.path.join(download_path, file),
                          os.path.join(datasets_folder, uuid_name + '.csv'))
            else:
                os.remove(os.path.join(download_path, file))

        # remove the dataset datasets_folder
        os.rmdir(download_path)

print("All CSV files downloaded and saved to DATASETS folder.")
