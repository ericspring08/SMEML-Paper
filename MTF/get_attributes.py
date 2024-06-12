import numpy as np
import pandas as pd
import os


def get_attributes(data_path):
    data = pd.read_csv(data_path)
    # Get the attributes of the data
    attributes = {}

    # Get the number of rows and columns
    attributes['num_rows'] = data.shape[0]
    print("Number of rows: ", data.shape[0])
    attributes['num_cols'] = data.shape[1]
    print("Number of columns: ", data.shape[1])

    # get the output distribution of 1 percentage
    attributes['output_distribution'] = max(data.iloc[:, -
                                                      1].value_counts(normalize=True))
    print("Output distribution: ", max(
        data.iloc[:, -1].value_counts(normalize=True)))

    # get percentage of numerical and categorical columns
    numerical_features = 0
    binary_categorical = 0
    categorical_average_count = 0
    categorical_count = 0
    iqr_values = []
    q1_values = []
    q3_values = []
    for col in data.columns:
        if data[col].nunique() > 15:
            numerical_features += 1

            # get the interquartile range
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            iqr_values.append(iqr)
            q1_values.append(q1)
            q3_values.append(q3)
        else:
            categorical_count += 1
            # categorical
            # check if binary or not
            if data[col].nunique() == 2:
                binary_categorical += 1

            categorical_average_count += data[col].nunique()

    attributes['numerical_columns'] = numerical_features/data.shape[1]
    attributes['binary_categorical'] = binary_categorical/categorical_count
    attributes['categorical_average_count'] = categorical_average_count / \
        categorical_count
    attributes['average_iqr'] = np.mean(iqr_values)
    attributes['average_q1'] = np.mean(q1_values)
    attributes['average_q3'] = np.mean(q3_values)
    attributes['iqr_std'] = np.mean(np.std(iqr_values))
    attributes['q1_std'] = np.mean(np.std(q1_values))
    attributes['q3_std'] = np.mean(np.std(q3_values))
    print("Numerical columns: ", numerical_features/data.shape[1])
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
    z_score_average = np.mean(np.abs((data - data.mean())/data.std()))
    z_score_std = np.mean(np.std(np.abs((data - data.mean())/data.std())))

    attributes['z_score_average'] = z_score_average
    print("Z-score average: ", z_score_average)
    attributes['z_score_std'] = z_score_std
    print("Z-score std: ", z_score_std)

    if data.shape[0] * data.shape[1] > 100000:
        attributes['average_correlation'] = np.nan
        attributes['correlation_std'] = np.nan
    else:
        # correlation matrix of the dataset
        corr_matrix = data.corr()

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
