import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data from 'heart.csv'
data = pd.read_csv('heart.csv')

# Separate features (X) and target variable (y)
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the task and the metric
task = Task('binary')  # binary classification
metric = 'roc_auc'  # use ROC AUC as the metric for binary classification tasks

# Create an instance of TabularAutoML
automl = TabularAutoML(task=task, timeout=600,
                       cpu_limit=4, reader_params={'n_jobs': 4})

# Fit the model and make predictions on the test data
predictions = automl.fit_predict(X_train, y_train, X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions.data[:, 0])
print(f'Mean Squared Error: {mse}')

# Calculate the accuracy
accuracy = (predictions.data[:, 0] > 0.5).mean()
print(f'Accuracy: {accuracy}')

# Save the model
automl.model.save('model.pkl')
