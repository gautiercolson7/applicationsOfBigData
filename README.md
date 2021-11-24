# Applications of Big Data Project

Home Credit Risk Project
Auteurs : Bilal LOUKILI, Gautier COLSON, Nadhem KHLIJ  

## Introduction

Le but de ce projet est de mettre en place des modèels de machine learning visant à prédire si un prêt peut être accordé à un individu, selon différents critères. Pour cela on s'appuie sur un jeu de données trai net u njeu de données train disponibles directement via la plateforme kaggle.  
La principale difficulté est de "nettoyer" le dataset, c'est à dire que de nombreuses colonnes sont inutiles ou alors des opérations sont à effectuer dessus. Les 3 modèles à créer sont XGBoost, Random Forest et Gradient Boosting

##Part 1
# Data exploration

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

# Feature engineering
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
Then we can work on the remaining columns

