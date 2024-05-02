# Summary of 1_Baseline

[<< Go back](../README.md)


## Baseline Classifier (Baseline)
- **n_jobs**: -1
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

0.2 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.692777 |  nan        |
| auc       | 0.5      |  nan        |
| f1        | 0.678663 |    0.461719 |
| accuracy  | 0.513619 |    0.461719 |
| precision | 0.513619 |    0.461719 |
| recall    | 1        |    0.461719 |
| mcc       | 0        |    0.461719 |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.692777 |  nan        |
| auc       | 0.5      |  nan        |
| f1        | 0.678663 |    0.461719 |
| accuracy  | 0.513619 |    0.461719 |
| precision | 0.513619 |    0.461719 |
| recall    | 1        |    0.461719 |
| mcc       | 0        |    0.461719 |


## Confusion matrix (at threshold=0.461719)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |                0 |              125 |
| Labeled as 1 |                0 |              132 |

## Learning curves
![Learning curves](learning_curves.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)



[<< Go back](../README.md)
