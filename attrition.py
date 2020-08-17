# In[1]:

import pickle
import pandas as pd
att = pd.read_csv(r"C:\Users\Tanmay Ambatkar\Documents\DataSets\Attrition.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
 
att.shape

att.columns

# shifting y variable to last 

att = att[['Age', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager','Attrition']]

att.head()

# removing EmployeeNumber

att.drop(['EmployeeNumber'],inplace = True,axis = 1)

# remove over18,standardhour,employeecount

att.drop(['Over18','EmployeeCount','StandardHours'],inplace = True,axis = 1)

# check the null values
att.isnull().sum()

att.shape
att.info()

# convert male to 1 and female to 0 
att.Gender.replace({"Male":1,"Female":0},inplace = True)

numcols = att.dtypes[att.dtypes != 'object'].index
numcols

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))
sns.heatmap(att[numcols].corr(),annot= True)

#check skewness of all numeric columns

from scipy.stats import kurtosis
from scipy.stats import skew


att_skewed = att[numcols].apply(lambda x:skew(x))
att_skewed.sort_values(ascending = False)  # sorting the values


# # Feature Selection
# now convert non numeric data to numeric
# using label encoder to turn factor into num

import sklearn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() # creating an object / instance for running for the first time

factcols = att.select_dtypes(include = 'object')
att[factcols.columns] = att[factcols.columns].apply(le.fit_transform)

att.head()

# # Select K-Best

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

kb = SelectKBest(score_func = chi2, k = "all")

#split the data

X = att.iloc[:,:-1]
Y = att.iloc[:,-1]
X.shape
Y.shape
fited = kb.fit(X,Y)

fited.scores_

feature_importance = pd.DataFrame({"Features":list(X.columns),"Importance":list(fited.scores_)})
# sorted
feature_importance.sort_values("Importance",ascending = False)

# Boruta 

from boruta import BorutaPy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
X1=np.array(X)

boruta_feature_selector=BorutaPy(rf,random_state=111,max_iter=25,perc=100,verbose=2)
boruta_feature_selector.fit(X1,Y)
boruta_feature_selector.support_

feature_importance2=pd.DataFrame({'Features':list(X.columns),'Importance':list(boruta_feature_selector.support_)})
feature_importance2

# RFE

from sklearn.feature_selection import RFE

rf = RandomForestClassifier()

rfe_rfc = RFE(rf,n_features_to_select = 12)

rfe_rfc.fit(X,Y)

rfe_rfc.support_

feature_importance3 = pd.DataFrame({"Features":list(X.columns),"Importance":list(rfe_rfc.support_)})
feature_importance3

# ## we have sorted the features by K-Best
# - Out of which we have seleted 12 features
# 1. MonthlyIncome
# 2. MonthlyRate
# 3. DailyRate
# 4. TotalWorkingYears
# 5. YearsAtCompany
# 6. NumCompaniesWorked
# 7. PercentSalaryHike 	
# 8. Age
# 9. OverTime
# 10. DistanceFromHome
# 11. HourlyRate
# 12. JobRole

# sampling

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X, Y,test_size = .2)

# random forest on all the data
 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 50)
train_x.shape
train_y.shape
rfc.fit(train_x,train_y)

pred_rf_all = rfc.predict(test_x)
from sklearn.metrics import confusion_matrix
tab3 = confusion_matrix(pred_rf_all, test_y)
a3 = tab3.diagonal().sum()*100/tab3.sum()
print(a3)
print(tab3)
from sklearn.metrics import classification_report
print(classification_report(pred_rf_all, test_y))

## plotting AUC Curve for all data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


att_auc_score = roc_auc_score(test_y, pred_rf_all) # gives me the AUROC value
att_auc_score

ns_probs = [0 for _ in range(len(test_y))]
ns_auc = roc_auc_score(test_y, ns_probs)

pred_value_pr = rfc.predict_proba(test_x)
pred_value_pr = pd.DataFrame(pred_value_pr)
pred_value_pr = pred_value_pr.iloc[:,-1]

fpr,tpr,threshold = roc_curve(test_y, pred_value_pr)

print('RandomForest: ROC AUC=%.3f' % (att_auc_score))
print('No Skills: ROC AUC=%.3f' % (ns_auc))

n_fpr , n_tpr, _ = roc_curve(test_y, ns_probs)

import matplotlib.pyplot as plt
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve on the Attrition on all data")
plt.plot(n_fpr, n_tpr, linestyle='--', label='No Skill')
plt.plot(fpr,tpr,color = "r", label = "RandomForest" % att_auc_score)
plt.legend(loc = "bottom right")

#precision recall curve on all data
 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

pred_value1 = rfc.predict(test_x)
pred_value_pr1 = rfc.predict_proba(test_x)
pred_value_pr2 = pd.DataFrame(pred_value_pr1)

precision, recall, thresholds = precision_recall_curve(test_y, pred_value_pr2[1])

f1_all = f1_score(test_y, pred_value1)
f1_all


plt.plot(recall, precision, label = "prec-recall(Area = %.2f)" % f1_all)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc = "upper left")
plt.title("Precision-Recall Curve")

# creating new data with best 12 features
att_new = att[['MonthlyIncome','MonthlyRate','DailyRate','TotalWorkingYears','YearsAtCompany',
               'NumCompaniesWorked','PercentSalaryHike','Age','OverTime','DistanceFromHome',
               'HourlyRate','JobRole','Attrition']]


att['Attrition'].value_counts()
att_new['JobRole'].value_counts()


att_new.shape
att_new.head()

# sampling

from sklearn.model_selection import train_test_split

att.shape
X1 = att_new.iloc[:,0:9]
X1.head()
Y1 = att_new.iloc[:,-1]
Y1.head()

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1,test_size = .2)

# random forest on selected data

from sklearn.ensemble import RandomForestClassifier
rfc_n = RandomForestClassifier(n_estimators = 50)

rfc_n.fit(X1_train,Y1_train)

pred_rf_new = rfc_n.predict(X1_test)
tab = confusion_matrix(pred_rf_new, Y1_test)
a = tab.diagonal().sum()*100/tab.sum()
print(a)
print(tab)
from sklearn.metrics import classification_report
print(classification_report(pred_rf_new, Y1_test))


## plotting AUC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


att_auc_score1 = roc_auc_score(Y1_test, pred_rf_new) # gives me the AUROC value
att_auc_score1

ns_probs1 = [0 for _ in range(len(Y1_test))]
ns_auc1 = roc_auc_score(Y1_test, ns_probs1)

pred_value_pr_new = rfc_n.predict_proba(X1_test)
pred_value_pr_new = pd.DataFrame(pred_value_pr_new)
pred_value_pr_new = pred_value_pr_new.iloc[:,-1]

fpr1,tpr1,threshold1 = roc_curve(Y1_test, pred_value_pr_new)

print('RandomForest: ROC AUC=%.3f' % (att_auc_score))
print('No Skills: ROC AUC=%.3f' % (ns_auc))

n_fpr1 , n_tpr1, _ = roc_curve(Y1_test, ns_probs1)

import matplotlib.pyplot as plt
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC Curve on the Attrition on feature selected data")
plt.plot(n_fpr1, n_tpr1, linestyle='--', label='No Skill')
plt.plot(fpr1,tpr1,color = "r", label = "RandomForest" % att_auc_score1)
plt.legend(loc = "bottom right")

#precision recall curve on feature selected data
 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

pred_value = rfc_n.predict(X1_test)
pred_value_pr = rfc_n.predict_proba(X1_test)
pred_value_pr1 = pd.DataFrame(pred_value_pr)

precision, recall, thresholds = precision_recall_curve(Y1_test, pred_value_pr1[1])

f1 = f1_score(Y1_test, pred_value)
f1


plt.plot(recall, precision, label = "prec-recall(Area = %.2f)" % f1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc = "upper left")
plt.title("Precision-Recall Curve")


# saving model to disk

pickle.dump(rfc,open('model.pkl','wb'))

# loading model to compare the results

model = pickle.load(open('model.pkl','rb'))



