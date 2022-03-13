import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

KPI = pd.read_csv('./KPI/train-data.csv')
x = list(KPI['value'])
y = list(KPI['label'])
period = 1650  #obtain from peak to peak estimation

mod = [i%period for i in range(len(x))] #binning the series by the period
num_iter = [math.floor(i/period) for i in range(len(x))] #nth period data is in used for differentiate the time
comb_x = list(zip(x,mod,num_iter))
    
x_train, x_test, y_train, y_test = train_test_split(comb_x, y, test_size=0.2, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

clf = AdaBoostClassifier()
clf.fit(x_train, y_train)

KPI_test = pd.read_hdf('./KPI/test-data.hdf')
X_test = list(KPI_test['value'])
Y_test = list(KPI_test['label'])
mod = [i%period for i in range(len(X_test))]
num_iter = [math.floor(i/period) for i in range(len(X_test))]
comb_x_test = list(zip(X_test,mod,num_iter))

test_predictions = logisticRegr.predict(comb_x_test)

print(classification_report(Y_test, test_predictions))

test_predictions = clf.predict(comb_x_test)

print(classification_report(Y_test, test_predictions))