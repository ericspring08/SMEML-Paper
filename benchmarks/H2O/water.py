import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import time

time_start = time.time()
h2o.init()

# Import a sample binary outcome train/test set into H2OAutoML
df = pd.read_csv("../../benchmark_datasets/heart.csv")

# Convert the pandas DataFrame to an H2OFrame.
h2o_df = h2o.H2OFrame(df)

# Identify predictors and response
x = h2o_df.columns

y = "target"

x.remove(y)

# For binary classification, response should be a factor
h2o_df[y] = h2o_df[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)

aml.train(x=x, y=y, training_frame=h2o_df)

# print accuracy
print(aml.leaderboard)

# print model
print(aml.leader)

time_end = time.time()

print("Time taken: ", time_end - time_start)
