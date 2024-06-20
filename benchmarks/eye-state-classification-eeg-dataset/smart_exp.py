import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from SMEML.SMEML import SMEML
import pandas as pd
import time
# import SMEML

time_start = time.time()

# Load the dataset
data = pd.read_csv('eye-state-classification-eeg-dataset.csv')

# Define the target column
target = data['eyeDetection']
data = data.drop(columns=['eyeDetection'])

if __name__ == '__main__':
    # Initialize the SMEML object
    smeml = SMEML(iterations=20, mode='SME')

    # Run the smart experiment
    smeml.train(data, target)

    time_end = time.time()

    print('Time elapsed: ', time_end - time_start)