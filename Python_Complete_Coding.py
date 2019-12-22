#!/usr/bin/env python
# coding: utf-8

# In[1]:


#---------<Step1: Prepare Data>--------------
# Data manipulation
import pandas as pd
import numpy as np
# Splitting data
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#Read Data
import pandas as pd
all_data = pd.read_csv("heloc_dataset.csv")
all_data.head()

#Test the data shape
print(all_data.shape)

#Transform Target Variable,'bad'=1, 'good'=0
dic = {'Bad':1,'Good':0}
all_data['RiskPerformance'] = all_data['RiskPerformance'].map(dic)
all_data.head()


# In[2]:


#-------<Step2: Data Cleaning>--------
# Deal with missing value (here -7,-8,-9)
# Checking missing data

#Deal with -9 (No Bureau Record or No Investigation)
missing_9=all_data.replace(-9,np.nan)
total = missing_9.isnull().sum().sort_values(ascending = False)
percent = (missing_9.isnull().sum()/missing_9.isnull().count()*100).sort_values(ascending = False)
missing_9  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("-----------Missing value for -9 is--------------\n",missing_9)

#Deal with -8 (No Usable/Valid Trades or Inquiries)
missing_8=all_data.replace(-8,np.nan)
total = missing_8.isnull().sum().sort_values(ascending = False)
percent = (missing_8.isnull().sum()/missing_8.isnull().count()*100).sort_values(ascending = False)
missing_8  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("\n-----------Missing value for -8 is--------------\n",missing_8)

#Deal with -7 (Condition not Met (e.g. No Inquiries, No Delinquencies))
missing_7=all_data.replace(-7,np.nan)
total = missing_7.isnull().sum().sort_values(ascending = False)
percent = (missing_7.isnull().sum()/missing_7.isnull().count()*100).sort_values(ascending = False)
missing_7  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("\n-----------Missing value for -7 is--------------\n",missing_7)


# In[4]:


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
after_7 = after_8.drop(['MSinceMostRecentDelq','MSinceMostRecentInqexcl7days'], axis=1)
after_7.head()


# In[5]:


# Use LabelEncoder to Deal with Category Variables (two columns)
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le1.fit_transform(after_7['MaxDelq2PublicRecLast12M'])
le2 = LabelEncoder()
le2.fit_transform(after_7['MaxDelqEver'])                      
used_data=after_7 


# In[6]:


#------<Step3: Model Selection>----------
from sklearn.metrics import roc_auc_score
data_prepared = used_data.drop(['RiskPerformance'], axis=1)
data_labels = used_data['RiskPerformance']

# Split into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(
    data_prepared,data_labels,random_state = 42)
a=train_labels.tolist()
a


# In[8]:


#-----<Model1: Decision Trees>---------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

tree_class = DecisionTreeClassifier()
tree_class.fit(train_features, train_labels)
scores = cross_val_score(tree_class, train_features, train_labels, scoring='roc_auc', cv=10)
tree_roc_scores = scores.mean()
print('The mean AUC for decision tree is:', tree_roc_scores)
tree_class.predict_proba(train_features)[:, 1]


# In[9]:


#Evaluate the best model on the test data
import sklearn
tree_class.fit(train_features, train_labels)
preds = tree_class.predict_proba(test_features)[:, 1]
baseline_auc1 = sklearn.metrics.roc_auc_score(test_labels, preds)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc1))


# In[10]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
from sklearn.model_selection import GridSearchCV
param_grid = [{'criterion':['gini'], 'max_depth':[2,3,4,5], 'min_samples_split':[3,5],'random_state':[1]}]
tree_class = DecisionTreeClassifier()
grid_search = GridSearchCV(tree_class, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc11 = sklearn.metrics.roc_auc_score(test_labels, preds)
print('The final tuned model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc11))


# In[11]:


#-----<Model2: Random Forest>---------
from sklearn.ensemble import RandomForestClassifier
#Create the basic model with 100 trees
forest_reg = RandomForestClassifier(n_estimators=50,
                               bootstrap = True,
                               max_features = 'sqrt')

RandomForest_scores = cross_val_score(forest_reg, train_features, train_labels, scoring='roc_auc', cv=10)
RandomForest_roc = RandomForest_scores.mean()
print('The mean ROC for Random Forest is:', RandomForest_roc)

forest_reg.fit(train_features, train_labels)

# Actual class predictions
rf_predictions = forest_reg.predict(test_features)
# Probabilities for each class
rf_probs = forest_reg.predict_proba(test_features)[:, 1]
rf_probs


# In[12]:


#Evaluate the best model on the test data
forest_reg.fit(train_features, train_labels)
preds = forest_reg.predict_proba(test_features)[:, 1]
baseline_auc2 = sklearn.metrics.roc_auc_score(test_labels, preds)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc2))


# In[13]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
              {'bootstrap':[False],'n_estimators':[3,10,20,30,40],'max_features':[2,3,4]}]
forest_reg = RandomForestClassifier()
grid_search = GridSearchCV(forest_reg, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc22 = sklearn.metrics.roc_auc_score(test_labels, preds)
print('The final tuned model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc22))


# In[14]:


#------<Model3: LightGBM>----------
# Split into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(data_prepared, data_labels,random_state = 42)

get_ipython().system('pip install lightgbm')
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.size'] = 18
get_ipython().run_line_magic('matplotlib', 'inline')

# Governing choices for search
N_FOLDS = 5
MAX_EVALS = 5


# First we can create a model with the default value of hyperparameters and score it using cross validation with early stopping. Using the cv LightGBM function requires creating a Dataset.
# Note: here we use **Automated Hyperparameter Tuning**

# In[15]:


#Establish a baseline model
base_LGBM = lgb.LGBMClassifier(random_state=50)
# Training set
train_set = lgb.Dataset(train_features, label = train_labels)
test_set = lgb.Dataset(test_features, label = test_labels)

# Default hyperparamters
hyperparameters = base_LGBM.get_params()
baseline_auc = cross_val_score(base_LGBM, train_features, train_labels, scoring='roc_auc', cv=10)
baseline_auc3= baseline_auc.mean()
print('The mean AUC for LGBM is:', baseline_auc3)

# Using early stopping to determine number of estimators.
del hyperparameters['n_estimators']

# Perform cross validation with early stopping
cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, metrics = 'auc', 
           early_stopping_rounds = 100, verbose_eval = False, seed = 42)

# Highest score
best = cv_results['auc-mean'][-1]

# Standard deviation of best score
best_std = cv_results['auc-stdv'][-1]


print('The maximium ROC AUC in cross validation was {:.5f} with std of {:.5f}.'.format(best, best_std))
print('The ideal number of iterations was {}.'.format(len(cv_results['auc-mean'])))


# In[16]:


#Evaluate the best model on the test data
#Optimal number of esimators found in cv
base_LGBM.n_estimators = len(cv_results['auc-mean'])

# Train and make predicions with model
base_LGBM.fit(train_features, train_labels)
preds = base_LGBM.predict_proba(test_features)[:, 1]
baseline_auc33 = roc_auc_score(test_labels, preds)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc33))


# In[17]:


#------<Model4: XGB>----------
import xgboost as xgb
#Establish a baseline model
base_xgb = xgb.XGBClassifier(random_state=50)

# Default hyperparamters
hyperparameters = base_xgb.get_params()
print(hyperparameters)

XGB_scores = cross_val_score(base_xgb, train_features, train_labels, scoring='roc_auc', cv=10)

print('The mean AUC for XGB is:', XGB_scores.mean())
base_xgb.fit(train_features, train_labels)

# Actual class predictions
xgb_predictions = base_xgb.predict(test_features)
# Probabilities for each class
base_xgb_probs = base_xgb.predict_proba(test_features)[:, 1]
base_xgb_probs
#AUC for test
baseline_auc4 = cross_val_score(base_xgb, test_features, test_labels, scoring='roc_auc', cv=10)


# In[18]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{'n_estimators':[10,50,100,150,200],'max_features':[2,4,6,8,10,12,14]},
              {'bootstrap':[False],'n_estimators':[10,50,100,150,200],'max_features':[2,3,4,5,6,7]}]
xgb_class = xgb.XGBClassifier(random_state=50)
grid_search = GridSearchCV(xgb_class, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc44 = roc_auc_score(test_labels, preds)
print('The final tuned XGB_model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc44))


# In[19]:


#-----<Model5: GaussianNB>-------
from sklearn.naive_bayes import GaussianNB

#Establish a baseline model
base_NB = GaussianNB()

# Default hyperparamters
hyperparameters = base_NB.get_params()
print(hyperparameters)

NB_scores = cross_val_score(base_NB, train_features, train_labels, scoring='roc_auc', cv=10)
print('The mean AUC for GaussianNB is:', NB_scores.mean())

base_NB.fit(train_features, train_labels)

# Actual class predictions
NB_predictions = base_NB.predict(test_features)
# Probabilities for each class
base_NB_probs = base_NB.predict_proba(test_features)[:, 1]
base_NB_probs
    
#AUC for test
baseline_auc5 = cross_val_score(base_NB, test_features, test_labels, scoring='roc_auc', cv=10)


# In[20]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{}]
NB_class = GaussianNB()
grid_search = GridSearchCV(NB_class, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc55 = roc_auc_score(test_labels, preds)
print('The final tuned NB_model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc55))


# In[21]:


#-----<Model6: Logistic Regression>-------
from sklearn.linear_model import LogisticRegression

#Establish a baseline model
base_LR = LogisticRegression()

# Default hyperparamters
hyperparameters = base_LR.get_params()
print(hyperparameters)

LR_scores = cross_val_score(base_LR, train_features, train_labels, scoring='roc_auc', cv=10)
print('The mean AUC for Logistic Regression is:', LR_scores.mean())

base_LR.fit(train_features, train_labels)

# Actual class predictions
LR_predictions = base_LR.predict(test_features)
# Probabilities for each class
base_LR_probs = base_LR.predict_proba(test_features)[:, 1]
base_LR_probs
#AUC for test
baseline_auc6 = cross_val_score(base_LR, test_features, test_labels, scoring='roc_auc', cv=10)    


# In[22]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{'C':[0.1,0.3,0.5,0.7,0.9,1], 'penalty':['l1','l2'],'random_state':[1]}]
base_LR = LogisticRegression()
grid_search = GridSearchCV(base_LR, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc66 = roc_auc_score(test_labels, preds)
print('The final tuned Logistic Regression model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc66))


# In[23]:


#-----<Model7: SVM>-------
from sklearn.svm import SVC

#Establish a baseline model
base_SVC = SVC(probability=True)

# Default hyperparamters
hyperparameters = base_SVC.get_params()
print(hyperparameters)

SVC_scores = cross_val_score(base_SVC, train_features, train_labels, scoring='roc_auc', cv=10)
print('The mean AUC for SVC Model is:', SVC_scores.mean())

base_SVC.fit(train_features, train_labels)

# Actual class predictions
SVC_predictions = base_SVC.predict(test_features)
# Probabilities for each class
base_SVC_probs = base_SVC.predict_proba(test_features)[:, 1]
base_SVC_probs

#AUC for test
baseline_auc7 = cross_val_score(base_SVC, test_features, test_labels, scoring='roc_auc', cv=10)    


# In[ ]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{'C':[0.1,0.3,0.5,0.7,0.9,1],'kernel':['rbf','linear'], 'max_iter':[-1],'random_state':[1]}]
base_SVC = SVC(probability=True)
grid_search = GridSearchCV(base_SVC, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)

grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc77 = roc_auc_score(test_labels, preds)
print('The final tuned SVC model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc77))


# In[ ]:


#------<Model8: KNN>----------
from sklearn.neighbors import KNeighborsClassifier as knn
#Establish a baseline model
base_knn = knn()

# Default hyperparamters
hyperparameters = base_knn.get_params()
print(hyperparameters)

knn_scores = cross_val_score(base_knn, train_features, train_labels, scoring='roc_auc', cv=10)
print('The mean AUC for KNN is:', knn_scores.mean())

base_knn.fit(train_features, train_labels)

# Actual class predictions
knn_predictions = base_knn.predict(test_features)
# Probabilities for each class
base_knn_probs = base_knn.predict_proba(test_features)[:, 1]
base_knn_probs
#AUC for test
baseline_auc8 = cross_val_score(base_knn, test_features, test_labels, scoring='roc_auc', cv=10)


# In[ ]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{},{'n_neighbors':[1,2,3,4,5,6,7,8,9,10],'weights':['uniform','distance'],'p':[1,2,3]}]
knn_class = knn()
grid_search = GridSearchCV(knn_class, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)
grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc88 = roc_auc_score(test_labels, preds)
print('The final tuned KNN_model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc88))


# In[ ]:


#------<Mode9: adaboost>----------
from sklearn.ensemble import AdaBoostClassifier as ada
#Establish a baseline model
base_ada = ada()

# Default hyperparamters
hyperparameters = base_ada.get_params()
print(hyperparameters)

ada_scores = cross_val_score(base_ada, train_features, train_labels, scoring='roc_auc', cv=10)
print('The mean AUC for AdaBoost is:', ada_scores.mean())
base_ada.fit(train_features, train_labels)

# Actual class predictions
ada_predictions = base_ada.predict(test_features)
# Probabilities for each class
base_ada_probs = base_ada.predict_proba(test_features)[:, 1]
base_ada_probs
#AUC for test
baseline_auc9 = cross_val_score(base_ada, test_features, test_labels, scoring='roc_auc', cv=10)


# In[ ]:


#----Hyper Parameter Tuning: Use GridSearchCV--------------
param_grid = [{'random_state':[1]},{'n_estimators':[10,20,30,40,50],'learning_rate':[0.1,0.5,1],'random_state':[1]}]
ada_class = ada()
grid_search = GridSearchCV(ada_class, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(train_features,train_labels)

cvres = grid_search.cv_results_ # the variable that stores the grid search results
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):  # iterate over the tested configurations
    print(mean_score, params)
grid_search.best_params_
grid_search.best_estimator_ # variable holding the best classifier (fitted on the entire dataset)

final_model = grid_search.best_estimator_

#Evaluate the best model on the test data
final_model.fit(train_features, train_labels)
preds = final_model.predict_proba(test_features)[:, 1]
baseline_auc99 = roc_auc_score(test_labels, preds)
print('The final tuned AdaBoost_model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc99))


# In[ ]:


#------<Step4: Record the Classifier Performance into a Comparison Table>--------
Model_Comparison={
            'Classifier':['Decision Trees','Random Forest','LightGBM','XGB',
                          'GaussianNB','Logistic Regression','SVM','KNN',
                          'AdaBoost'],
            'test_AUC':[baseline_auc1,baseline_auc2,baseline_auc3,baseline_auc4,
      baseline_auc5,baseline_auc6,baseline_auc7,baseline_auc8,baseline_auc9],
            'test_after_tuned_AUC':[baseline_auc11,baseline_auc22,baseline_auc33,baseline_auc44,
      baseline_auc55,baseline_auc66,baseline_auc77,baseline_auc88,baseline_auc99]
        }
Model_Comparison=pd.DataFrame(Model_Comparison,index=[0])
Model_Comparison=Model_Comparison.sort_values(by=['Classifier']).reset_index(drop=True)
Model_Comparison


# In[ ]:




