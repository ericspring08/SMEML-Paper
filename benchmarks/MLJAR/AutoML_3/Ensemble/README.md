# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model             |   Weight |
|:------------------|---------:|
| 4_Default_Xgboost |        5 |

## Metric details
|           |    score |     threshold |
|:----------|---------:|--------------:|
| logloss   | 0.006057 | nan           |
| auc       | 1        | nan           |
| f1        | 1        |   0.0342756   |
| accuracy  | 1        |   0.0342756   |
| precision | 1        |   0.0342756   |
| recall    | 1        |   1.94288e-06 |
| mcc       | 1        |   0.0342756   |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.006057 | nan         |
| auc       | 1        | nan         |
| f1        | 1        |   0.0342756 |
| accuracy  | 1        |   0.0342756 |
| precision | 1        |   0.0342756 |
| recall    | 1        |   0.0342756 |
| mcc       | 1        |   0.0342756 |


## Confusion matrix (at threshold=0.034276)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              125 |                0 |
| Labeled as 1 |                0 |              132 |

## Learning curves
![Learning curves](learning_curves.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)



[<< Go back](../README.md)
