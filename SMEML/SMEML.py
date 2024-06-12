import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import os

classifier_names = [
    'SVC',
    'SGDClassifier',
    'RidgeClassifierCV',
    'RidgeClassifier',
    'Perceptron',
    'PassiveAggressiveClassifier',
    'LogisticRegressionCV',
    'LogisticRegression',
    'LinearSVC',
    'RandomForestClassifier',
    'HistGradientBoostingClassifier',
    'GradientBoostingClassifier',
    'ExtraTreesClassifier',
    'AdaBoostClassifier',
    'XGBClassifier',
    'LGBMClassifier',
    'CatBoostClassifier',
    'RadiusNeighborsClassifier',
    'KNeighborsClassifier',
    'NearestCentroid',
    'QuadraticDiscriminantAnalysis',
    'LinearDiscriminantAnalysis',
    'GaussianNB',
    'BernoulliNB',
    'MLPClassifier',
    'ExtraTreeClassifier',
    'DecisionTreeClassifier',
    'LabelSpreading',
    'LabelPropagation',
    'DummyClassifier'
]


class SMEML:
    def __init__(self):
        self.X = None
        self.y = None

    def train(self, X, y):
        self.X = X
        self.y = y
        attributes = self.get_dataset_attributes()

        # train the model
        smeml_model = pkl.load(
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl'), 'rb'))

        input = list(attributes.values())
        input = np.array(input).reshape(1, -1)

        results = smeml_model.predict(input)

        top_3 = np.argsort(results[0])[-3:]

        print("Top 3 classifiers: ", [classifier_names[i] for i in top_3])

        return results, attributes

    def get_dataset_attributes(self):
        attributes = {}

        attributes['num_rows'] = self.X.shape[0]
        print("Number of rows: ", self.X.shape[0])
        attributes['num_cols'] = self.X.shape[1]
        print("Number of columns: ", self.X.shape[1])

        # get the output distribution of 1 percentage
        attributes['output_distribution'] = max(self.X.iloc[:, -
                                                            1].value_counts(normalize=True))
        print("Output distribution: ", max(
            self.X.iloc[:, -1].value_counts(normalize=True)))

        # get percentage of numerical and categorical columns
        numerical_features = 0
        binary_categorical = 0
        categorical_average_count = 0
        categorical_count = 0
        iqr_values = []
        q1_values = []
        q3_values = []
        for col in self.X.columns:
            if self.X[col].nunique() > 15:
                numerical_features += 1

                # get the interquartile range
                q1 = self.X[col].quantile(0.25)
                q3 = self.X[col].quantile(0.75)
                iqr = q3 - q1

                iqr_values.append(iqr)
                q1_values.append(q1)
                q3_values.append(q3)
            else:
                categorical_count += 1
                # categorical
                # check if binary or not
                if self.X[col].nunique() == 2:
                    binary_categorical += 1

                categorical_average_count += self.X[col].nunique()

        attributes['numerical_columns'] = numerical_features/self.X.shape[1]
        attributes['binary_categorical'] = binary_categorical/categorical_count
        attributes['categorical_average_count'] = categorical_average_count / \
            categorical_count
        attributes['average_iqr'] = np.mean(iqr_values)
        attributes['average_q1'] = np.mean(q1_values)
        attributes['average_q3'] = np.mean(q3_values)
        attributes['iqr_std'] = np.mean(np.std(iqr_values))
        attributes['q1_std'] = np.mean(np.std(q1_values))
        attributes['q3_std'] = np.mean(np.std(q3_values))
        print("Numerical columns: ", numerical_features/self.X.shape[1])
        print("Binary categorical: ", binary_categorical/categorical_count)
        print("Categorical average count: ",
              categorical_average_count/categorical_count)
        print("Average IQR: ", np.mean(iqr_values))
        print("Average Q1: ", np.mean(q1_values))
        print("Average Q3: ", np.mean(q3_values))
        print("IQR STD: ", np.mean(np.std(iqr_values)))
        print("Q1 STD: ", np.mean(np.std(q1_values)))
        print("Q3 STD: ", np.mean(np.std(q3_values)))

        # average percentage of values with a z-score greater than 3 in each dataset
        z_score_average = np.mean(
            np.abs((self.X - self.X.mean())/self.X.std()))
        z_score_std = np.mean(
            np.std(np.abs((self.X - self.X.mean())/self.X.std())))

        attributes['z_score_average'] = z_score_average
        print("Z-score average: ", z_score_average)
        attributes['z_score_std'] = z_score_std
        print("Z-score std: ", z_score_std)

        if self.X.shape[0] * self.X.shape[1] > 100000:
            attributes['average_correlation'] = np.nan
            attributes['correlation_std'] = np.nan
        else:
            # correlation matrix of the dataset
            corr_matrix = self.X.corr()

            # get the average correlation of the dataset (excluding the diagonal values)
            corr_values = []

            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    if i != j:
                        corr_values.append(corr_matrix.iloc[i, j])

            attributes['average_correlation'] = np.mean(corr_values)
            print("Average correlation: ", np.mean(corr_values))
            attributes['correlation_std'] = np.mean(np.std(corr_values))
            print("Correlation std: ", np.mean(np.std(corr_values)))

        return attributes
