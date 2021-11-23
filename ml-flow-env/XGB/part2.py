import sys
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost
import mlflow.sklearn



csv_path=r"C:\Users\LOUKILI\Desktop\app-big-data\big data project\data\application_train.csv"
train = pd.read_csv(csv_path)
csv_path=r"C:\Users\LOUKILI\Desktop\app-big-data\big data project\data\application_test.csv"
test = pd.read_csv(csv_path)


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

test = test.drop(columns=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 
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
				   
				   



train = train.drop(columns = ['NAME_TYPE_SUITE', 'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_HOUR',
					 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
					 'AMT_REQ_CREDIT_BUREAU_YEAR']);

test = test.drop(columns = ['NAME_TYPE_SUITE', 'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_HOUR',
					 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
					 'AMT_REQ_CREDIT_BUREAU_YEAR']);



train = train.drop(columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE'])
test = test.drop(columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE'])
train_copy = train[['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']].copy()
test_copy = test[['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']].copy()

train_contact_type = pd.get_dummies(train_copy['NAME_CONTRACT_TYPE'])
train_gender = pd.get_dummies(train_copy['CODE_GENDER'])
frames = [train_copy, train_contact_type, train_gender]
train_final = pd.concat(frames, axis = 1)
train_final = train_final.drop(columns = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER'])

test_contact_type = pd.get_dummies(test_copy['NAME_CONTRACT_TYPE'])
test_gender = pd.get_dummies(test_copy['CODE_GENDER'])
frames = [test_copy, test_contact_type, test_gender]
test_final = pd.concat(frames, axis = 1)
test_final = test_final.drop(columns = ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER'])

X = train_final[['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'Cash loans', 'Revolving loans', 'F', 'M', 'XNA']]
y = train_final['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


def main():
       
    ## Allow us to parse value for the model when launching the script    
    eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    colsample = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    subsample = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    # enable auto logging
    mlflow.xgboost.autolog()
    
    # Set mlflow experiment name
    mlflow.set_experiment("XGBOOST")
    
    ## Start mlflow run
    with mlflow.start_run():
        
        ## Define the model
        XGB = XGBClassifier(objective='binary:logistic', eval_metric="logloss", use_label_encoder=False, eta=eta, subsample=subsample
                           , colsample_bytree=colsample)
        ## Train the model
        XGB.fit(X_train, y_train)

        # Make prediction and then evaluate the model with accuracy, precision and confusion matrix
        y_pred = XGB.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test,y_pred)
        prec = precision_score(y_test, y_pred)
               
        ## Make confusion matrix into a png to store it    
        hm = sb.heatmap(cm,annot=True, fmt='g')
        plt.savefig('hm.png')
        mlflow.log_artifact("hm.png")
        os.remove('hm.png')
        
        ## log metrics and choosen parameters
        mlflow.log_metrics({"accuracy": acc,"precision": prec})
        mlflow.log_param("eta", eta)
        mlflow.log_param("colsample_bytree", colsample)
        mlflow.log_param("subsample", subsample)
        
        ## Store the model
        mlflow.sklearn.log_model(XGB, "XGB_model")

if __name__ == "__main__":
    main()