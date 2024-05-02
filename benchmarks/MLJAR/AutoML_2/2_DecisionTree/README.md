# Summary of 2_DecisionTree

[<< Go back](../README.md)


## Decision Tree
- **n_jobs**: -1
- **criterion**: gini
- **max_depth**: 3
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

6.9 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.38683  |  nan        |
| auc       | 0.898485 |  nan        |
| f1        | 0.849624 |    0.42     |
| accuracy  | 0.844358 |    0.42     |
| precision | 0.901235 |    0.864407 |
| recall    | 1        |    0        |
| mcc       | 0.688436 |    0.42     |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.38683  |      nan    |
| auc       | 0.898485 |      nan    |
| f1        | 0.849624 |        0.42 |
| accuracy  | 0.844358 |        0.42 |
| precision | 0.843284 |        0.42 |
| recall    | 0.856061 |        0.42 |
| mcc       | 0.688436 |        0.42 |


## Confusion matrix (at threshold=0.42)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              104 |               21 |
| Labeled as 1 |               19 |              113 |

## Learning curves
![Learning curves](learning_curves.png)

## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)



[<< Go back](../README.md)
