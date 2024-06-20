import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..')))
print(sys.path)
# import SMEML
from SMEML.SMEML import SMEML
import pandas as pd
import time

time_start = time.time()

# Load the dataset
data = pd.read_csv('pima-indians-diabetes-database.csv')

# Define the target column
target = data['Outcome']
data = data.drop(columns=['Outcome'])

if __name__ == '__main__':
    # Initialize the SMEML object
    smeml = SMEML(iterations=20, mode='dumb')

    # Run the smart experiment
    smeml.train(data, target)

    time_end = time.time()

    print('Time elapsed: ', time_end - time_start)