##################################################
#Author : Team Invincible Predictors
#Project : Recruit Restaurant Visitor Forecasting
#Please make sure that this script is run where all CSV files are stored.
#################################################
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Importing CSV Files...")

#Importing CSV Files
air_reserve = pd.read_csv('air_reserve.csv',parse_dates=['visit_datetime', 'reserve_datetime'])
air_store_info = pd.read_csv('air_store_info.csv')
store_id_relation = pd.read_csv('store_id_relation.csv')
date_info = pd.read_csv('date_info.csv',parse_dates=['calendar_date'])
train = pd.read_csv('train.csv',parse_dates=['visit_date'])

print("Importing CSV Files... Done")

print("Feature Engineering in progress...")

#dropping the day of week column as we would already extract it from datetime value.
date_info.drop(columns=['day_of_week'],inplace=True)

#renaming date column so that it will be useful while merging the holiday flag with training data.
date_info.rename(columns={'calendar_date':'visit_date'},inplace=True)

#preparing final training dataset by merging relevant features to the train data.
train_data = train.merge(air_store_info, how='left', on='air_store_id')

#Extracting year month weekday as new features
train_data["visit_year"] = pd.DatetimeIndex(train_data['visit_date']).year
train_data["visit_month"] = pd.DatetimeIndex(train_data['visit_date']).month
train_data["visit_weekday"] = pd.DatetimeIndex(train_data['visit_date']).weekday

#Extracting city ward neighbourhood as new features
train_data['city'] = train_data['air_area_name'].str.split().str[0]
train_data['ward'] = train_data['air_area_name'].str.split().str[1]
train_data['neighborhood'] = train_data['air_area_name'].str.split().str[2]

#Add holiday flag from date info table
train_data = train_data.merge(date_info,how='left',on='visit_date')

#Making all object type columns as categorical columns.
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category')
        
#Implementing labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in ['city','ward','neighborhood','holiday_flg','air_genre_name','air_area_name']:
    train_data[feature] = le.fit_transform(train_data[feature].astype(str))

    
# #Implementing OneHotEncoding using get dummies method
train_data = pd.concat([train_data,pd.get_dummies(train_data['holiday_flg'], prefix='holiday_flg')],axis=1)
train_data.drop(['holiday_flg'],axis=1, inplace=True)     

#Adding mean median and min max visitors column grouped by air store id and weekday
group_by_cols = ['air_store_id','visit_weekday']
visitor_stats = train_data\
                .groupby(group_by_cols)\
                ['visitors']\
                .agg(['mean', 'median', 'min','max'])\
                .rename(columns=lambda colname: str(colname)+'_visitors')\
                .reset_index()

train_data = train_data.merge(visitor_stats,how='left',on=group_by_cols)

print("Feature Engineering completed...")

# Create evaluation function (the competition uses Root Mean Square Log Error)
from sklearn.metrics import mean_squared_log_error

def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

print("Train Test split in progress...")


#train test split
from sklearn.model_selection import train_test_split
X = train_data.drop(["air_store_id","visit_date","visitors","air_area_name","longitude"], axis=1)
y = train_data["visitors"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Train Test split Done...")


print("Testing out different models...")
print("Linear Regression Test in Progress...")
#Trying simple Linear Regression model

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_preds=lr_model.predict(X_test)
print("Linear Regression RMSLE Score : ",rmsle(y_test, y_preds))

print("KNeighbors Regression Test in Progress...")
#Trying KNeighbors Regression model

from sklearn.neighbors import KNeighborsRegressor
knr_model = KNeighborsRegressor(n_jobs=-1, n_neighbors=10)
knr_model.fit(X_train, y_train)
y_preds=knr_model.predict(X_test)
print("KNeighbors Regression RMSLE Score : ",rmsle(y_test, y_preds))


print("Random Forest Regression Test in Progress...")
#Trying Random Forest Regressor Regression model 

from sklearn.ensemble import RandomForestRegressor

rfrmodel = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                 min_samples_split=15,
                                 max_features=1, n_jobs=-1, 
                                 )

rfrmodel.fit(X_train, y_train)
y_preds=rfrmodel.predict(X_test)
rmsle(y_test, y_preds)
print("Random Forest Regression RMSLE Score : ",rmsle(y_test, y_preds))

print("XGBoost Regression Test in Progress...")
#Trying XGBoost Regression model

from xgboost import XGBRegressor
xgbmodel = XGBRegressor(
                        max_depth =16,
                        learning_rate=0.1, 
                        n_estimators=20, 
                        subsample=0.4, 
                        colsample_bytree=0.8,
                        seed=5
                       )
xgbmodel.fit(X_train,y_train)

y_preds_xgb=xgbmodel.predict(X_test)
print("xgboost Regression RMSLE Score : ",rmsle(y_test,y_preds_xgb))


print("Applying feature engineering on actual test data...")

#Performing exact same operations, applied on train data, for the sample submission data

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['air_store_id'] = sample_submission['id'].str.rsplit('_',1).str[0]
sample_submission['visit_date'] = sample_submission['id'].str.rsplit('_',1).str[1]
sample_submission.visit_date = pd.to_datetime(sample_submission.visit_date)
sample_submission["visit_year"] = pd.DatetimeIndex(sample_submission['visit_date']).year
sample_submission["visit_month"] = pd.DatetimeIndex(sample_submission['visit_date']).month
sample_submission["visit_weekday"] = pd.DatetimeIndex(sample_submission['visit_date']).weekday
sample_submission = sample_submission.merge(air_store_info, how='left', on='air_store_id')
sample_submission.drop(columns=['id'],inplace=True)
sample_submission['city'] = sample_submission['air_area_name'].str.split().str[0]
sample_submission['ward'] = sample_submission['air_area_name'].str.split().str[1]
sample_submission['neighborhood'] = sample_submission['air_area_name'].str.split().str[2]
sample_submission = sample_submission.merge(date_info,how='left',on='visit_date')

#Making all object type columns as categorical columns.
for col in sample_submission.columns:
    if sample_submission[col].dtype == 'object':
        sample_submission[col] = sample_submission[col].astype('category')

#Implementing labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in ['city','ward','neighborhood','holiday_flg','air_genre_name','air_area_name']:
    sample_submission[feature] = le.fit_transform(sample_submission[feature].astype(str))

sample_submission = pd.concat([sample_submission,pd.get_dummies(sample_submission['holiday_flg'], prefix='holiday_flg')],axis=1)
sample_submission.drop(['holiday_flg'],axis=1, inplace=True)

sample_submission=sample_submission[['air_store_id', 'visit_date', 'visitors', 'air_genre_name',
       'air_area_name','latitude', 'longitude','visit_year', 'visit_month', 'visit_weekday', 'city',
       'ward', 'neighborhood', 'holiday_flg_0', 'holiday_flg_1']]

sample_submission = sample_submission.merge(visitor_stats,how='left',on=group_by_cols)
sample_submission = sample_submission.fillna(train_data.mean())

print("Feature engineering on actual test data completed...")

#Specifying the training and test data, here test data is our predictions in sample submission.
X_train = train_data.drop(["air_store_id","visit_date","visitors","air_area_name","longitude"], axis=1)
Y_train = train_data["visitors"]

X_test = sample_submission.drop(["air_store_id","visit_date","visitors","air_area_name","longitude"], axis=1)
y_test = sample_submission["visitors"]

final_submission = pd.read_csv('sample_submission.csv')

print("Applying Linear Regression...")

#Modelling Linear regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
y_preds_lr=lr_model.predict(X_test)
final_submission['visitors']=y_preds_lr
final_submission.to_csv('prediction_lr.csv', index=False)

print("Completed, please check prediction_lr.csv")

print("Applying KNeighbors Regression...")
#Modelling KNeighbors regression
from sklearn.neighbors import KNeighborsRegressor
knr_model = KNeighborsRegressor(n_jobs=-1, n_neighbors=10)
knr_model.fit(X_train, Y_train)
y_preds_knr=knr_model.predict(X_test)
final_submission['visitors']=y_preds_knr
final_submission.to_csv('prediction_knr.csv', index=False)
print("Completed, please check prediction_knr.csv")

print("Applying Random Forest Regression...")
#Modelling Random Forest regression
from sklearn.ensemble import RandomForestRegressor

rfrmodel = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                 min_samples_split=15,
                                 max_features=1, n_jobs=-1, 
                                 )

rfrmodel.fit(X_train, Y_train)
y_preds_rfr=rfrmodel.predict(X_test)
final_submission['visitors']=y_preds_rfr
final_submission.to_csv('prediction_rfr.csv', index=False)
print("Completed, please check prediction_rfr.csv")

print("Applying XGBoost Regression...")
#Trying XGBoost Regression model

from xgboost import XGBRegressor
xgbmodel = XGBRegressor(
                        max_depth =16,
                        learning_rate=0.1, 
                        n_estimators=20, 
                        subsample=0.4, 
                        colsample_bytree=0.8,
                        seed=5
                       )
xgbmodel.fit(X_train,Y_train)
y_preds_xgb=xgbmodel.predict(X_test)
final_submission['visitors']=y_preds_xgb
final_submission.to_csv('prediction_xgb.csv', index=False)
print("Completed, please check prediction_xgb.csv")