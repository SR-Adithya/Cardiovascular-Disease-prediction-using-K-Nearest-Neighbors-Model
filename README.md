# Cardiovascular-Disease-prediction-using-SVM
Exposure to cardiovascular disease by collecting clinical parameters and processing through a machine learning model.

## About Dataset
Name: Cardiovascular Study Dataset. Columns Used: age, sex, is_smoking, cigsPerday, sysBP, diaBP, heartRate. No of Rows: 3390 records

## Model Selection and it's reason:
Model: K-Nearest Neighbor Classification (Supervised Model)
Description:
•	The K-Nearest Neighbor classifier was selected because cardiovascular disease exposure results in categorical groups.
•	It works on idea of proximity based learning, in which new individual’s risk category is calculated comparing their clinical profile. Thus supporting for detecting trends in health data
•	Cardiovascular risk assessment is influenced by several features. Hence, distance based KNN classification is appropriate for capturing trends between clinical features.

## Training method of data:
•	Total Dataset used = 3390 records
•	Test size were listed and used test_size = 0.20. Hence, 678 records were adopted as test case and 2712 records were assigned in training the model.
•	Scaler used: StandardScaler. Utilized to normalize several features into a single value.
•	To assist the model with learning multiple possibilities, random_state 42 was implemented.

## Metrics used for Evaluation:
Metrics used:
		Confusion matrix: 
	Confusion matrix represents a table visual that highlights number of predictions made correct and incorrect for each class by the model
	The values are count based representation and used for other evaluation metrics.
	| **Predicted \\ Actual** | **Positive (1)** | **Negative (0)** |
|-------------------------|------------------|------------------|
| **Positive (1)**        | TP               | FP               |
| **Negative (0)**        | FN               | TN               |


**Where:** 
-**\(TP (True positive )\)**=correct prediction of positive classes
-**\(TN (True negative )\)**=correct prediction of negative classes
-**\(FP (False positive )\)**=incorrect prediction of positive classes
-**\(FN (False negative )\)**= incorrect prediction of negative classes
	
True positive (TP) and true negatives (TN) provide the perfect right as right and perfect wrong as wrong prediction classes whereas false positive (FP) and false negative (FN) values show where the model predicts the right ones as wrong and vice versa.
Accuracy score: 
Used to measure the closeness of the measured value to the standard value. A single value that summarizes the whole model’s performance
The value is calculated by using the below formula:

$$
\text{Accuracy score} = \frac{TP+TN}{TP+TN+FP+FN}
$$

**Where:** 
-**\(TP (True positive )\)**=correct prediction of positive classes
-**\(TN (True negative )\)**=correct prediction of negative classes
-**\(FP (False positive )\)**=incorrect prediction of positive classes
-**\(FN (False negative )\)**= incorrect prediction of negative classes
	
The higher value of accuracy score marks the model’s performance at its best. Low accuracy score shows the model struggles to differentiate between classes. 
Classification report: 
Provides comprehensive performance analysis such as recall, precision, F1 score and shows the behavior of each classes.
Recall: Ratio of correct predicted positive among all actual positives
Formula used: 
$$
\text{Recall} = \frac{TP}{(TP+FN)}
$$
Helps to capture exact positive cases

Precision: Ratio of correct predicted positives among all predicted positives
Formula used:
$$
\text{Precision} = \frac{TP}{(TP+FP)}
$$
Useful to mitigate false positives

F1-Score: Utilizes precision and recall and calculates their mean harmonically to balance the false positives and negatives into a single value.
Formula used:
$$
\text{F1 score} = \frac{2 × (Precision × Recall)}{(Precision + Recall)}
$$
Reduces false positives and false negatives

Evaluated Metric Values from the model:

Confusion matrix: 
	| **Predicted \\ Actual** | **Positive (1)** | **Negative (0)** |
|-------------------------|------------------|------------------|
| **Positive (1)**        | 10               | 25               |
| **Negative (0)**        | 94               | 549               |

  The contingency table utilized 678 test case record
	549 records were correctly predicted as negative (healthy) case, 25 false positive (unhealthy), 94 false negative (healthy), and 10 as correctly predicted as positive case (unhealthy)
	Hence, the model should be optimized to reduce false positive and false negative

Accuracy score = 0.82

Classification report:
Recall: 0.96 (for healthy), 0.10 (for cardiovascular disease exposure)
Precision: 0.85 (for healthy), 0.29 (for cardiovascular disease exposure)
F1 score: 0.90 (for healthy), 0.14 (for cardiovascular disease exposure)

## Strengths and Weakness of the model:
Strength: The model identifies non-linear correlations between numerous clinical characteristics and performs well without making any assumptions about the underlying data distribution.
Weakness: KNN's computational cost rises with dataset size, and feature scaling and K value selection have an impact on performance.

## Possible improvements / real-world applications:
Improvement: To enhamce generalisation and lessen susceptibility to noise, use cross-validation to optimise the value of K. Examine KNN's performance in comparison to other classifiers and investigate ensemble methods for increased resilience.

real-world application: Fast forwarding cardiac cases for early prevention of cardiovascular disease, provide balanced diet for healthy life. Telemedicine platforms or hospital information systems can incorporate health monitoring measures, programs for health assessment at community level, educational resources to raise patient understanding of cardiovascular risk due to physiological and lifestyle variables.

## Conclusion of the project:
From an individual’s new clinical parameter data, the trained K-Nearest Neighbor classifier has successfully predicted cardiovascular disease. The model had the highest performance with a promising classification report and the best projected accuracy of 0.82. Additionally, when significantly more data is encountered, this would show that the model might be used in a real-time setting for further model research.
