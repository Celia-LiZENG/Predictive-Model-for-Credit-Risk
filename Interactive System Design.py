#-----<Step1: Intstall Required Packages>---------
import streamlit as st
import pickle
import numpy as np
from sklearn import metrics
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
#!pip install lightgbm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Title and Subheader
st.title("Credit Risk Evaluation - Your Best Helper")
st.subheader("Machine Learning based App with Streamlit ")

# Load the pipeline and data
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))

dic = {0: 'bad', 1: 'good'} 
pipe1 = pickle.load(open('pipe_LR.sav', 'rb'))
pred1 = pipe1.predict_proba(X_test)
score1 = pipe1.score(X_test, y_test)
auc1 = cross_val_score(pipe1, X_test, y_test, scoring='roc_auc', cv=10)
auc1=auc1.mean()
pipe2 = pickle.load(open('pipe_LGBM.sav', 'rb'))
pred2 = pipe2.predict_proba(X_test)
score2 = pipe2.score(X_test, y_test)
auc2 = cross_val_score(pipe2, X_test, y_test, scoring='roc_auc', cv=10)
auc2=auc2.mean()
    
pipe3 = pickle.load(open('pipe_XGB.sav', 'rb'))
pred3 = pipe3.predict_proba(X_test)
score3 = pipe3.score(X_test, y_test)
auc3 = cross_val_score(pipe3, X_test, y_test, scoring='roc_auc', cv=10)
auc3=auc3.mean()
clf_compare ={'Accuracy':[score1,score2,score3],
                     'AUC Score':[auc1,auc2,auc3]}
clf_compare = pd.DataFrame(clf_compare)
clf_compare.index=['Logistic Regression','Light GBM','XGBoost']


if st.checkbox('Show dataframe --->>>'):
    st.write(X_test.head())

#Function to test certain index of dataset
def test_demo(index):
    values = X_test.iloc[index] 
    # Create User Input Data in the sidebar/text
    ExternalRiskEstimate = st.sidebar.slider("External Risk Estimate",0.0,100.0,values[0],1.0)
    MSinceOldestTradeOpen = st.sidebar.slider("Months since Oldest Trade Open",0.0,1000.0,values[1],1.0)
    MSinceMostRecentTradeOpen = st.sidebar.slider("Months since Most Recent Trade Open",0.0,500.0,values[2],1.0)
    AverageMInFile = st.sidebar.slider("Average Months in File",0.0,500.0,values[3],1.0)
    NumSatisfactoryTrades = st.sidebar.slider("Number of Satisfy Trades",0.0,80.0,values[4],1.0)
    NumTrades60Ever2DerogPubRec = st.sidebar.slider("Number Trades 60+ Ever",0.0,50.0,values[5],1.0)
    NumTrades90Ever2DerogPubRec = st.sidebar.slider("Number Trades 90+ Ever",0.0,50.0,values[6],1.0)
    PercentTradesNeverDelq = st.sidebar.slider("Percent Trades Never Delinquent",0.0,100.0,values[7],1.0)
    MaxDelq2PublicRecLast12M= st.sidebar.slider("Percent Trades Never Delinquent",0.0,9.0,values[8],1.0)
    MaxDelqEver = st.sidebar.slider("Percent Trades Never Delinquent",1.0,9.0,values[9],1.0)
    NumTotalTrades = st.sidebar.slider("Number of Total Trades", 0.0,150.0,values[10],1.0)
    NumTradesOpeninLast12M = st.sidebar.slider("Enter the Number of Trades Open in Last 12 Months",0.0,150.0,values[11],1.0)
    PercentInstallTrades = st.sidebar.slider("Percent Installment Trades",1.0,100.0,values[12],1.0)
    NumInqLast6M = st.sidebar.slider("Number of Inq Last 6 Months",0.0,100.0,values[13],1.0)
    NumInqLast6Mexcl7days= st.sidebar.slider("Number of Inq Last 6 Months excl 7days", 0.0,100.0,values[14],1.0)
    NetFractionRevolvingBurden = st.sidebar.slider("Net Fraction Revolving Burden",0.0,300.0,values[15],1.0)
    NumRevolvingTradesWBalance = st.sidebar.slider("Number Revolving Trades with Balance",0.0,50.0,values[16],1.0)
    NumInstallTradesWBalance= st.sidebar.slider("Number Installment Trades with Balance",0.0,100.0,values[17],1.0)
    NumBank2NatlTradesWHighUtilization= st.sidebar.slider("Number Bank/Natl Trades w high utilization ratio",0.0,100.0,values[18],1.0)
    PercentTradesWBalance= st.sidebar.slider("Percent Trades with Balance",0.0,100.0,values[19],1.0)

    #Print the prediction result
    if st.checkbox('See the model performance comparison before choosing your favorite one --->>>'):
        st.write(clf_compare)
        st.write('               the comparison of 3 Best Models      ')
        st.write(st.line_chart(data=clf_compare))
    alg = ['Light GBM', 'XGBoost', 'Logistic Regression']
    classifier = st.selectbox('Which model do you want to choose for prediction?', alg)
    if classifier == 'Logistic Regression':
        pipe1 = pickle.load(open('pipe_LR.sav', 'rb'))
        user_prediction_data = np.array([ExternalRiskEstimate, MSinceOldestTradeOpen,
       MSinceMostRecentTradeOpen, AverageMInFile, NumSatisfactoryTrades,
       NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec,
       PercentTradesNeverDelq, MaxDelq2PublicRecLast12M,MaxDelqEver,NumTotalTrades, NumTradesOpeninLast12M,
       PercentInstallTrades, NumInqLast6M, NumInqLast6Mexcl7days,
       NetFractionRevolvingBurden, NumRevolvingTradesWBalance,
       NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization,
       PercentTradesWBalance]).reshape(1,20) 
        
        res = pipe1.predict_proba(user_prediction_data)
        st.write('According to your input, the default risk is predicted to be', res[:,1])
        pred = pipe1.predict_proba(X_test)
        score = pipe1.score(X_test, y_test)
        #cm = metrics.confusion_matrix(y_test, pred)
        auc = cross_val_score(pipe1, X_test, y_test, scoring='roc_auc', cv=10)
        auc=auc.mean()
        st.text('for Logistic Regression Model Chosen, you have:')
        st.write('Accuracy: ', score)
        st.write('AUC Score: ', auc)
        #st.write('Confusion Matrix: ', cm)
        
    
    elif classifier == 'Light GBM':
        pipe2 = pickle.load(open('pipe_LGBM.sav', 'rb'))
        
        user_prediction_data = np.array([ExternalRiskEstimate, MSinceOldestTradeOpen,
       MSinceMostRecentTradeOpen, AverageMInFile, NumSatisfactoryTrades,
       NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec,
       PercentTradesNeverDelq, MaxDelq2PublicRecLast12M,MaxDelqEver,NumTotalTrades, NumTradesOpeninLast12M,
       PercentInstallTrades, NumInqLast6M, NumInqLast6Mexcl7days,
       NetFractionRevolvingBurden, NumRevolvingTradesWBalance,
       NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization,
       PercentTradesWBalance]).reshape(1,20) 
        
        res = pipe2.predict_proba(user_prediction_data)
        st.write('According to your input, the default risk is predicted to be', res[:,1])
        pred = pipe2.predict_proba(X_test)
        score = pipe2.score(X_test, y_test)
        #cm = metrics.confusion_matrix(y_test, pred)
        auc = cross_val_score(pipe2, X_test, y_test, scoring='roc_auc', cv=10)
        auc=auc.mean()
        st.text('for Light GBM Model Chosen, you have')
        st.write('Accuracy: ', score)
        st.write('AUC Score: ', auc)
        #st.write('Confusion Matrix: ', cm)
        
        
    elif classifier == 'XGBoost':
        pipe3 = pickle.load(open('pipe_XGB.sav', 'rb'))
        
        user_prediction_data = np.array([ExternalRiskEstimate, MSinceOldestTradeOpen,
       MSinceMostRecentTradeOpen, AverageMInFile, NumSatisfactoryTrades,
       NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec,
       PercentTradesNeverDelq, MaxDelq2PublicRecLast12M,MaxDelqEver,NumTotalTrades, NumTradesOpeninLast12M,
       PercentInstallTrades, NumInqLast6M, NumInqLast6Mexcl7days,
       NetFractionRevolvingBurden, NumRevolvingTradesWBalance,
       NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization,
       PercentTradesWBalance]).reshape(1,20) 
        
        res = pipe3.predict_proba(user_prediction_data)
        st.write('According to your input, the default risk is predicted to be', res[:,1])
        pred = pipe3.predict_proba(X_test)
        score = pipe3.score(X_test, y_test)
        #cm = metrics.confusion_matrix(y_test, pred)
        auc = cross_val_score(pipe3, X_test, y_test, scoring='roc_auc', cv=10)
        auc=auc.mean()
        st.text('for XGBoost Model Chosen, you have:')
        st.write('Accuracy: ', score)
        st.write('AUC Score: ', auc)
        #st.write('Confusion Matrix: ', cm)
        
    
    
number = st.text_input('Choose a row of information in the dataset (0~119):', 1)          
test_demo(int(number))  # Run the test function
