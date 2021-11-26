ML Models used
===================

Now that we made changes on our dataset, we can build and train our models easily. We have 3 models to build : XGBoost, Random Forest and Gradient Boosting.<br>

XGBoost
------------

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.  
So first we need to import all the required library, especially scikit-learn. We can then setup 2 variables (X, y) to store our data and our target.  

.. code-block:: python
	from xgboost import XGBClassifier
	from sklearn.model_selection import train_test_split, cross_val_score
	from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
	from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

	X = train_final[['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'Cash loans', 'Revolving loans', 'F', 'M', 'XNA']]
	y = train_final['TARGET']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
	# fit model no training data
	model = XGBClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print(accuracy)

Random Forest
------------

Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
We also need to import the RandomForestClassifier module from sklearn, train and predict our model.

.. code-block:: python
	from sklearn.ensemble import RandomForestClassifier

	model = RandomForestClassifier()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	print("Training Accuracy :", model.score(X_train, y_train))
	print("Testing Accuracy :", model.score(X_test, y_test))

Gradient Boosting
------------

Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting.  
Same process here, we are importing the RandomForestClassifier module, train and test our model then.<br><br>

**Conclusion :**  
------------

By testing our models, we see that our accuracy for the 3 models are above 0.9, it is satisfying for this exercise, keep in mind that the accuracy of our models are not the main topic here. We are considering this as a decent score for a model very easy to manipulate, with very few columns, regarding the initial datase.
