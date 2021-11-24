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



