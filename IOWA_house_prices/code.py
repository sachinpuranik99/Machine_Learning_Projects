from infra import *

main_file_path = 'train.csv'
test_file_path = 'test.csv'
data = pd.read_csv(main_file_path)
test_data = pd.read_csv(test_file_path)

#Final predictions 
predictors = ['PoolArea','GrLivArea','YrSold','KitchenAbvGr','MiscVal',
        'OverallCond', 'OverallQual','LotArea', 'YearBuilt', '1stFlrSF',
        '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
        'TotRmsAbvGrd','MSZoning','Street', 'Alley', 'LotShape', 'LandContour',
         'LandSlope' , 'Neighborhood', 'BldgType']
         #, 'MSSubClass','Utilities','HouseStyle', 'ExterQual','Foundation',
         #'CentralAir','Functional']
         #'LotFrontage','LotConfig',,'RoofStyle',,'RoofMatl','Exterior1st']#,'GarageArea']

#Train data
train_data = data[predictors]
modified_train_data= train_data.copy()
#Convert cat attributes to the numerical/one-hot using get_dummies function
modified_train_data = pd.get_dummies(modified_train_data)
#Missing column information
cols_with_missing = modified_train_data.columns[train_data.isnull().any()]  
y = data.SalePrice

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

#Imputing the missing values
final_train_data = my_imputer.fit_transform(final_train_data)
final_test_data = my_imputer.fit_transform(final_test_data)


my_model = XGBRegressor(n_estimators = 4000, learning_rate = 0.06)
my_model.fit(final_train_data, y, verbose= False)
test_y = my_model.predict(final_test_data)
sub_csv = pd.DataFrame({'Id': test_data.Id, 'SalePrice':test_y})
sub_csv.to_csv('final_submission.csv', index=False)
