import argparse as ap
import os
import pandas as pd
import matplotlib.pyplot as plt

parser = ap.ArgumentParser()

parser.add_argument('-f', help='Folder')

args = parser.parse_args()

folder = args.f

# go through all the files in the folder
for parent_folder in os.listdir(folder):
    # check if folder has file dataset.csv
    if not os.path.exists(os.path.join(folder, parent_folder, 'dataset.csv')):
        continue
    print(f'Converting {parent_folder}')
    df = pd.read_csv(os.path.join(folder, parent_folder, 'dataset.csv'))

    plt.imsave(os.path.join(folder, parent_folder,
               'dataset.png'), df.values, cmap='gray')
