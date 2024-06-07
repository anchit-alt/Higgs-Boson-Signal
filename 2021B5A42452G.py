import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import zipfile
train_df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train_df.info())
print(test.info())
#NO NULL VALUES
y = train_df['Target']
X = train_df.drop('Target',axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0)
#Plotting the graphs for features vs Target
plt.scatter(x='Feat-'+str(5), y='Target', data=train_df)
# plt.show()
plt.scatter(x='Feat-'+str(9), y='Target', data=train_df)
# plt.show()
#Calculating Correlation of the features with the target column
i = 1
corr_list = []
while i<31:
 corr = train_df.Target.corr(train_df["Feat-"+str(i)])
 corr_list.append(corr)
 i+=1
print(corr_list)
#dropping Feat - {2,8,17,21} as they are very badly correlated with the
# target respectively
train_df.drop(train_df.columns[[1, 7, 16,20]], axis=1, inplace=True)
input_columns = list(train_df.columns)[1:-1]
print(input_columns)
target_col = 'Target'
train_targets = train_df[target_col].copy()
train_inputs = train_df[input_columns].copy()
test_targets = train_df[target_col].copy()
test_inputs = test[input_columns].copy()
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
print(train_df)
print(train_inputs)
print(test_inputs)
X_train = train_inputs[numeric_cols]
test_inputs = test[input_columns].copy()
ID = test['ID']
X_test = test_inputs[numeric_cols]
from sklearn.ensemble import GradientBoostingClassifier
classifier =GradientBoostingClassifier(learning_rate=0.3,n_estimators=100,max_depth=5)
classifier.fit(X_train, train_targets)
test_pred = classifier.predict((X_test))
print(classifier.classes_)
cm = confusion_matrix(y_test,test_pred,labels = classifier.classes_)
print(cm)
cm_disp =ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["0","1"])
cm_disp.plot()
plt.show()
df_submission = pd.DataFrame({'ID': ID, 'Labels': test_pred})
df_submission.to_csv('submission.csv', index=False)
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write('submission.csv')
    archive.write('eval.ipynb')
os.remove('submission.csv')