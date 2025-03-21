import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import time
import pandas as pd
from SMEML.SMEML import SMEML
print(sys.path)
# import SMEML

time_start = time.time()

# Load the dataset
data = pd.read_csv('brain-stroke-prediction-dataset.csv')

# Define the target column
target = data['stroke']
data = data.drop(columns=['stroke'])

if __name__ == '__main__':
    # Initialize the SMEML object
    smeml = SMEML(iterations=20, mode='SME')

    # Run the smart experiment
    smeml.train(data, target)

    time_end = time.time()

    print('Time elapsed: ', time_end - time_start)
