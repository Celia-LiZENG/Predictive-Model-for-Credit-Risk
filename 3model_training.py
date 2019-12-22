#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
get_ipython().system('pip install lightgbm')
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Data Processing
all_data = pd.read_csv("heloc_dataset.csv")
all_data.head()
dic = {'Bad':1,'Good':0}
all_data['RiskPerformance'] = all_data['RiskPerformance'].map(dic)

# for -9: the distribution is uniform among all the attributes, and the proportion is relatively small --> delete all the rows
after_9=all_data.replace(-9,np.nan)
after_9.dropna(axis=0, how='any', inplace=True)

# for -8: delete the 'NetFractionInstallBurden' column; the remained use imputer to fill the NA
after_8_del = after_9.drop(['NetFractionInstallBurden'], axis=1)
after_8=after_8_del.replace(-8,np.nan)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
after_8=imp.fit_transform(after_8)
after_8=pd.DataFrame(after_8)
after_8.columns=after_8_del.columns

# for -7: delete the 'MSinceMostRecentDelq' and 'MSinceMostRecentInqexcl7days'
# didn't use fill_value method because those variables mean: NOT satisfy the condition --> NOT traditional miss the data?
from sklearn.preprocessing import LabelEncoder
after_7 = after_8.drop(['MSinceMostRecentDelq','MSinceMostRecentInqexcl7days'], axis=1)
after_7.head()
le1 = LabelEncoder()
le1.fit_transform(after_7['MaxDelq2PublicRecLast12M'])
le2 = LabelEncoder()
le2.fit_transform(after_7['MaxDelqEver'])                      
used_data=after_7                        

#--------split Data into train+test----------
data_prepared = used_data.drop(['RiskPerformance'], axis=1)
data_labels = used_data['RiskPerformance']
X_train, X_test, y_train, y_test = train_test_split(data_prepared,data_labels,random_state = 42)


#--------Three Best Models Picked --------------
# Create a pipeline to fit the data
#Best Model1: lightGBM
from sklearn.model_selection import KFold
pipe_LGBM = Pipeline([('minmax', MinMaxScaler()), 
                          ('LGBM', LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=29, n_jobs=-1, num_leaves=31, objective=None,
               random_state=50, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0))])

pipe_LGBM.fit(X_train, y_train)


# Save the data and pipeline
pickle.dump(X_train, open('X_train.sav', 'wb'))
pickle.dump(pipe_LGBM, open('pipe_LGBM.sav', 'wb'))
pickle.dump(X_test, open('X_test.sav', 'wb'))
pickle.dump(y_test, open('y_test.sav', 'wb'))

#-----Best Model2: XGB----------
pipe_XGB = Pipeline([('minmax', MinMaxScaler()), 
                          ('LXGB', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3, max_features=2,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=50,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1))])

pipe_XGB.fit(X_train, y_train)
pickle.dump(pipe_XGB, open('pipe_XGB.sav', 'wb'))


#--------Best Model3: Logistic Regression--------
pipe_LR = Pipeline([('minmax', MinMaxScaler()), 
                          ('LR', LogisticRegression(C=0.7, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=1, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False))])

pipe_LR.fit(X_train, y_train)
pickle.dump(pipe_LR, open('pipe_LR.sav', 'wb'))









# In[ ]:




