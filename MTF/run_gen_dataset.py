import os
import argparse as ap
import subprocess

parser = ap.ArgumentParser(description='Generate datasets')

parser.add_argument('-f',
                    '--folder',
                    type=str,
                    help='Folder containing datasets',
                    required=True)

parser.add_argument('-o',
                    '--output',
                    type=str,
                    help='Output folder',
                    required=True)

args = parser.parse_args()

datasets_folder = args.folder
output_folder = args.output

# read the datasets folder
datasets = os.listdir(datasets_folder)

for index, dataset in enumerate(datasets):
    if dataset.endswith('.csv'):
        print(f"[{index}] Generating {dataset}")
        # print subprocess output
        result = subprocess.run(
            f"python3 gen_dataset.py -i {datasets_folder}/{dataset} -o {os.path.join(output_folder, dataset.split('.')[0])}", shell=True, capture_output=True)
        print(result.stdout.decode('utf-8'))
