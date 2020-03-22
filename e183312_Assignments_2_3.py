import pandas as pd

df = pd.read_csv("ESCP-Train.csv")

#Inspecting the data.
df.describe()
df.isnull().sum()

#Checking correlation with dependent variable.
df.corr()[1:2]

#The variables purchaseTime, visitTime, hour are dropped due to high correlation; id is also dropped.
df2 = df.drop(['id', 'purchaseTime', 'visitTime','hour'], axis=1)

X = df2.loc[:, df2.columns != "label"]
y = df2.loc[:, df2.columns == "label"]

#Test and train. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 0, stratify = y)

#Oversampling.
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = 'minority', random_state = 123)
X_train, y_train = sm.fit_resample(X_train, y_train)

#Random forest.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=123, n_estimators=60)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict_proba(X_test)
y_pred_rfc = rfc.predict(X_test)

print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_rfc))

from sklearn.model_selection import cross_val_score
from numpy import mean
scores = cross_val_score(rfc, X, y, scoring='roc_auc',n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))

#Testing the model.
df_test = pd.read_csv("ESCP-Test.csv")

df_test.head()

del df_test['label']

df_test = df_test.drop(['id', 'purchaseTime', 'visitTime','hour'], axis=1)

prediction = pd.DataFrame(rfc.predict_proba(df_test), columns=['Prob_0','Prob_1'])
df_test['Prob_0'] = prediction['Prob_0']
df_test['Prob_1'] = prediction['Prob_1']

Result = pd.DataFrame()

Result['Prob_0'] = df_test['Prob_0']

Result['Prob_1'] = df_test['Prob_1']

Result.to_csv("Results_of_Prediction.csv")



