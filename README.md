# Applications of Big Data Project

Home Credit Risk Project
Auteurs : Bilal LOUKILI, Gautier COLSON, Nadhem KHLIJ  

## Introduction

Le but de ce projet est de mettre en place des modèels de machine learning visant à prédire si un prêt peut être accordé à un individu, selon différents critères. Pour cela on s'appuie sur un jeu de données trai net u njeu de données train disponibles directement via la plateforme kaggle.  
La principale difficulté est de "nettoyer" le dataset, c'est à dire que de nombreuses colonnes sont inutiles ou alors des opérations sont à effectuer dessus. Les 3 modèles à créer sont XGBoost, Random Forest et Gradient Boosting

## Part 1
### Data exploration

```python
df.shape
```
To see the shape of the dataset : 307511 rows and 122 columns. <br><br>  

```python
train.info(max_cols = 130)
```
To see the name of each column, the type and the number of non-null cells.<br><br>

```python
pd.set_option("display.max_columns",None)
train.head(10)
```
To see the first 10 rows of data, with all the columns. It help to see what can be se values for each feature.<br><br>

```python
target = train.TARGET
count = target.value_counts()
percentage = target.value_counts(normalize = True)*100
pd.DataFrame({'counts': count, 'percentage' : percentage})
```
![1](https://user-images.githubusercontent.com/70965407/143295455-a4bceacc-4beb-450a-9371-17fe17e8c0c9.PNG)  
To visualize the split of target value = 1 and target value = 0<br><br>

```python
plt.hist(combi['CODE_GENDER'], bins = 5, color = 'blue')
plt.title('Male and Female loan applicants')
plt.xlabel('Gender')
plt.ylabel('population')
plt.show()
```
![2](https://user-images.githubusercontent.com/70965407/143296121-573f3b35-3915-4013-b4f4-1fecd3f1ad1a.PNG)  
To visualize the repartition of gender<br><br>

```python
plt.hist(combi['NAME_FAMILY_STATUS'], bins = 5, color = 'brown')
plt.title('Marraige Statu loan applicants')
plt.xlabel('Marraige Status')
plt.ylabel('population')
plt.show()
```
![3](https://user-images.githubusercontent.com/70965407/143296350-ef7edde3-51af-4bf1-b0db-d00dcdaba0aa.PNG)  
To show the repartition of the family status  

### Feature engineering
The main purpose of the project is not to have the more performant model, but to understand how to undustrialize a Machine Learning project with different tools like MLFlow. So due to this, we are going to reduce the number of column, to keep the ones we think are the more relevant. We are aware that by doing this, our model is going to be less efficient, but easier to manipulate.

```python
train = train.drop(columns=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 
                   'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11',
                   'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                   'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 
                   'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
                   'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
                   'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
                   'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
                   'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                   'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
                   'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
                   'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                   'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'])
```
So here basically we are deleting the columns the less important from our point of view, based on the data exploration. Most of theses columns are just irrelevant regarding the target, have very few values, or a disturbing median.
Then we can work on the remaining columns.

```python
print(train.isnull().sum())
```
Here we visualize the number of null cells for each column.  <br><br>
![4](https://user-images.githubusercontent.com/70965407/143323752-1032fba3-9abd-4497-a5e7-17e999b91ac1.PNG)  
We see that 4 columns have 1021 null values so we can also delete these columns, too much null values to have a good impact on the target.
```python
train = train.drop(columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE'])
```
<br>
Finally we are going to transform litteral values into numerical ones. With this part of code :  <br><br>

```python
train_contact_type = pd.get_dummies(train_copy['NAME_CONTRACT_TYPE'])
train_gender = pd.get_dummies(train_copy['CODE_GENDER'])
frames = [train_copy, train_contact_type, train_gender]
train_final = pd.concat(frames, axis = 1)
train_final = train_final.drop(columns = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER'])
train_final.head()
```
We also drop the unique ID column which is useless for a machine learning model, and the columns with litteral valeus because now we have them in numerical values. Let's print the head of the dataset now.<br><br>
![5](https://user-images.githubusercontent.com/70965407/143325883-8af58db3-b974-4c4a-9dac-cac9b4fb7935.PNG)  

### Modeling
Now that we made changes on our dataset, we can build and train our models easily. We have 3 models to build : XGBoost, Random Forest and Gradient Boosting.<br>

**XGBoost**  
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.  
So first we need to import all the required library, especially scikit-learn. We can then setup 2 variables (X, y) to store our data and our target.  
```python
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
```
<br>

**Random Forest**  
Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
We also need to import the RandomForestClassifier module from sklearn, train and predict our model.
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training Accuracy :", model.score(X_train, y_train))
print("Testing Accuracy :", model.score(X_test, y_test))
```  
<br>

**Gradient Boosting**  
Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting.  
Same process here, we are importing the RandomForestClassifier module, train and test our model then.<br><br>

**Conclusion :**  
By testing our models, we see that our accuracy for the 3 models are above 0.9, it is satisfying for this exercise, keep in mind that the accuracy of our models are not the main topic here. We are considering this as a decent score for a model very easy to manipulate, with very few columns, regarding the initial datase.  

## Part 2 : MLFlow

Now we are going to execute our python code without all the visuals, with a .py format to allow us tu use it from command line. After our models training, we can open the MLFlow UI, which will allow us to visualize the performance of our model with different parameters.  
![6](https://user-images.githubusercontent.com/70965407/143507333-f5ca8bda-edd1-469f-8696-d5a62bf20008.png)
<br>  
![7](https://user-images.githubusercontent.com/70965407/143507348-1d95161c-bd0c-4440-b02b-edbfedb8fe5e.png)
<br><br>
We can now choose the artifact with the best prediction results that was generated by mlflow for the prediction. You can set up the artifact with the model serve command.   
![8](https://user-images.githubusercontent.com/70965407/143507723-a80bbc3a-62a0-4c2c-984e-c1144b5420a5.png)
<br><br>
The model is now available for prediction. We use the following command, passing values for our used columns. We note here all the usefulness of the feature engineering work, which makes it easier for us to put into production. The model returns us either 0 or 1 depending on its prediction.
```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\":[\"AMT_CREDIT\",\"AMT_INCOME_TOTAL\", \"Cash loans\", \"Revolving loans\", \"F\", \"M\", \"XNA\"],\"data\":[[406597.5, 202500.0, 1, 0, 0, 1, 0]]}" http://localhost:1234/invocations
```
