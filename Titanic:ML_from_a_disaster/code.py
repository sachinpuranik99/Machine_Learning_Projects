import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import Imputer 
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

main_file_path = 'train.csv'
test_file_path = 'test.csv'
data = pd.read_csv(main_file_path)
test_data = pd.read_csv(test_file_path)

#Final predictions 
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']

#Train data
train_data = data[predictors]
modified_train_data= train_data.copy()
#Convert cat attributes to the numerical/one-hot using get_dummies function
modified_train_data = pd.get_dummies(modified_train_data)
#Missing column information
cols_with_missing = modified_train_data.columns[train_data.isnull().any()]  
y = data.Survived

#Test data
#test_data_original = test_data[predictors]
test_data_original = test_data[train_data.columns]
modified_test_data = test_data_original.copy()
modified_test_data = pd.get_dummies(modified_test_data)
cols_with_missing = modified_test_data.columns[test_data_original.isnull().any()]  

#Add another columns with indication of data missing or not.
for col in cols_with_missing:
    modified_train_data[col + "_is_missing"] = modified_train_data[col].isnull()
    modified_test_data[col + "_is_missing"] = modified_test_data[col].isnull()

#Aligning as per the column names
final_train_data, final_test_data = modified_train_data.align(modified_test_data,
                                                              join='left', 
                                                              axis=1)
my_imputer = Imputer()

#Imputing the missing values
final_train_data = my_imputer.fit_transform(final_train_data)
final_test_data = my_imputer.fit_transform(final_test_data)


ran_forest_model = RandomForestClassifier()
ran_forest_model.fit(final_train_data, y)
test_y = ran_forest_model.predict(final_test_data)
#my_model = XGBClassifier(n_estimators = 1000, learning_rate = 0.1)
#my_model.fit(final_train_data, y, verbose= False)
#test_y = my_model.predict(final_test_data)
sub_csv = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':test_y})
sub_csv.to_csv('final_submission.csv', index=False)
