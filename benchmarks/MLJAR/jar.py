import pandas as pd
from supervised import AutoML

# Load the dataset
data = pd.read_csv('heart.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns=['target'])
y = data['target']

# Initialize AutoML
automl = AutoML()

# Train the model
automl.fit(X, y)

# Evaluate the model (optional)
score = automl.score(X, y)
print("Model Score:", score)
