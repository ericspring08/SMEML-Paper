import h2o
from h2o.estimators import H2ORandomForestEstimator

h2o.init()

h2o_data = h2o.import_file('heart.csv')

train, test = h2o_data.split_frame(ratios=[0.8])

x = train.columns[:-1]
y = 'target'

rf_model = H2ORandomForestEstimator()
rf_model.train(x=x, y=y, training_frame=train)

predictions = rf_model.predict(test)

print(rf_model)
print(rf_model.model_performance(test))

h2o.cluster().shutdown()
