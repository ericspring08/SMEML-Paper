import os
from kaggle.api.kaggle_api_extended import KaggleApi
import glob
import shutil

api = KaggleApi()

api.authenticate()

datasets = []
start_page = 1
page_count = 10
topics = ['disease', 'covid', 'coronavirus',
          'virus', 'infection', "cancer", 'heart', 'lungs']
for i in range(start_page, start_page + page_count + 1):
    for j in range(len(topics)):
        dataset_page = api.dataset_list(
            search=topics[j], file_type='csv', max_size=10000000, page=i)
        datasets.extend(dataset_page)

datasets_folder = 'DATASETS'
os.makedirs(datasets_folder, exist_ok=True)

for index, dataset in enumerate(datasets):
    dataset_name = dataset.ref.split('/')[-1]
    download_path = os.path.join(datasets_folder, dataset_name)

    # check if the dataset is already downloaded
    if os.path.exists(os.path.join(datasets_folder, str(dataset.id) + '.csv')):
        print(f"[{index}] {dataset_name} {dataset.id} already downloaded")
        continue
    else:
        print(f"[{index}] Downloading {dataset_name} {dataset.id}")

    if not os.path.exists(download_path):
        api.dataset_download_files(
            dataset.ref, path=download_path, quiet=True, unzip=True)
        print(f"[{index}] Downloaded {dataset_name} {dataset.id}")

        # go through the files in the dataset folder
        for file in glob.glob(download_path + '/*'):
            # check if the file is a CSV file
            if file.endswith('.csv'):
                # move the file to the datasets_folder
                print(f"[{index}] Moving {file} to {datasets_folder}")
                os.rename(file, os.path.join(
                    datasets_folder, str(dataset.id) + '.csv'))

        # remove the dataset datasets_folder
        shutil.rmtree(download_path, ignore_errors=True)

print("All CSV files downloaded and saved to DATASETS folder.")
