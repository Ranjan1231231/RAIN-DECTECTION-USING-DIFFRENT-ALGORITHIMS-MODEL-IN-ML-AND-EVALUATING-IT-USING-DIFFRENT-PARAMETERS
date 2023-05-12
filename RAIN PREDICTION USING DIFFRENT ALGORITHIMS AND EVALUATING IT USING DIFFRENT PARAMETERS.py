import pandas as pd
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
df=pd.read_csv('Weather_Data.csv')
# print(df.head(),df.shape,df.size)
#DATA PREPROCESSING
df_sydney_processed=pd.get_dummies(data=df,columns=['RainToday','WindGustDir','WindDir9am','WindDir3pm'])
df_sydney_processed.replace(['No','Yes'],[0,1],inplace=True)
#Training data and test data
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed=df_sydney_processed.astype(float)
features=df_sydney_processed.drop(columns='RainTomorrow',axis=1)
Y=df_sydney_processed['RainTomorrow']

#TRAIN/TEST SPLIT
x_train,x_test,y_train,y_test=train_test_split(features,Y,test_size=0.2,random_state=10)


#LINEAR REGRESSION MODEL
LinearReg=LinearRegression()
LinearReg.fit(x_train,y_train)
LINR_predictions=LinearReg.predict(x_test)
# print(predictions)
LinearRegression_MAE=metrics.mean_absolute_error(y_test,LINR_predictions)
LinearRegression_MSE=metrics.mean_squared_error(y_test,LINR_predictions)
LinearRegression_R2=metrics.r2_score(y_test,LINR_predictions)
# print(LinearRegression_R2,LinearRegression_MAE,LinearRegression_MSE)
LR_Report=pd.DataFrame({'Metric':['LINEAR REGRESSION MAE','LINEAR REGRESSION MSE','LINEAR REGRESSION R2'],'Value':[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2]})
# print(LR_Report)
#KNN MODEL
KNN=KNeighborsClassifier(n_neighbors=4).fit(x_train,y_train)
KNN_predictions=KNN.predict(x_test)
# print(KNN_predictions)
KNN_Accuracy_Score=metrics.accuracy_score(y_test,KNN_predictions)
KNN_JaccardIndex=metrics.jaccard_score(y_test,KNN_predictions)
KNN_F1_Score=metrics.f1_score(y_test,KNN_predictions)
# print(KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score)
KNN_Report=pd.DataFrame({'Metric':['KNN Accuracy Score','KNN Jaccard Index','KNN F1 Score'],'Value':[KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score]})
# print(KNN_Report)


#DECISION TREE
Decision_Tree=DecisionTreeClassifier()
Decision_Tree.fit(x_train,y_train)
DT_predictions=Decision_Tree.predict(x_test)
# print(DT_predictions)
DT_Accuracy_Score=metrics.accuracy_score(y_test,DT_predictions)
DT_Jaccard_Index=metrics.jaccard_score(y_test,DT_predictions)
DT_F1_Score=metrics.f1_score(y_test,DT_predictions)
DT_Report=pd.DataFrame({'Metric':['DT Accuracy Score','DT Jaccard Index','DT F1 Score'],'Value':[DT_Accuracy_Score,DT_Jaccard_Index,DT_F1_Score]})
# print(DT_Report)


#LOGISTIC REGRESSION
x_train,x_test,y_train,y_test=train_test_split(features,Y,test_size=0.2,random_state=1)
LOGR=LogisticRegression(solver='liblinear')
LOGR.fit(x_train,y_train)
LOGR_predictions=LOGR.predict(x_test)
LOGR_predicion_probability=LOGR.predict_proba(x_test)
# print(LOGR_predictions)
# print(LOGR_predicion_probability)
LOGR_Accuracy_Score=metrics.accuracy_score(y_test,LOGR_predictions)
LOGR_Jaccard_Index=metrics.jaccard_score(y_test,LOGR_predictions)
LOGR_F1_Score=metrics.f1_score(y_test,LOGR_predictions)
LOGR_Log_Loss=metrics.log_loss(y_test,LOGR_predictions)
LOGR_Report=pd.DataFrame({'Metric':['Logistic Regression Accuracy Score','Logistic Regression Jaccard Index','Logistic Regression F1 Score','Logistic Regression Log loss'],'Value':[LOGR_Accuracy_Score,LOGR_Jaccard_Index,LOGR_F1_Score,LOGR_Log_Loss]})
# print(LOGR_Report)


#SUPPORT VECTOR MACHINE
SVM=svm.SVC()
SVM.fit(x_train,y_train)
SVM_Predictions=SVM.predict(x_test)
SVM_Accuracy_Score = metrics.accuracy_score(y_test,SVM_Predictions)
SVM_Jaccard_Index = metrics.jaccard_score(y_test,SVM_Predictions)
SVM_F1_Score = metrics.f1_score(y_test,SVM_Predictions)
SVM_Report=pd.DataFrame({'Metric':['SVM Accuracy Score','SVM Jaccard Index','SVM F1 Score'],'Value':[SVM_Accuracy_Score,SVM_Jaccard_Index,SVM_F1_Score]})
# print(SVM_Report)

#TOTAL REPORT ACCURACY,JACCARD INDEX,F1 SCORE, LOG LOSS
#LINEAR REGRESSION
LinearRegression_LogLoss=metrics.log_loss(y_test,LINR_predictions)
#KNN
KNN_Log_Loss=metrics.log_loss(y_test,KNN_predictions)
#DECISION TREE
DT_Log_Loss=metrics.log_loss(y_test,DT_predictions)
#SVM
SVM_Log_Loss=metrics.log_loss(y_test,SVM_Predictions)

TOTAL_REPORT=pd.DataFrame({'METHOD':['Accuracy_Score','Jaccard_Index','F1_Score','LogLoss'],'LINEAR REGRESSION MODEL':[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2,LinearRegression_LogLoss],'K NEAREST NEIGHBOUR MODEL':[KNN_Accuracy_Score,KNN_JaccardIndex,KNN_F1_Score,KNN_Log_Loss],'DECISION TREE MODEL':[DT_Accuracy_Score,DT_Jaccard_Index,DT_F1_Score,DT_Log_Loss], 'LOGISTIC REGRESSION MODEL':[LOGR_Accuracy_Score,LOGR_Jaccard_Index,LOGR_F1_Score,LOGR_Log_Loss],'SUPPORT VECTOR MACHINE':[SVM_Accuracy_Score,SVM_Jaccard_Index,SVM_F1_Score,SVM_Log_Loss]})
print(TOTAL_REPORT.to_csv('dsv.csv'))