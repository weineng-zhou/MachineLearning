# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:32:10 2020

@author: weineng.zhou
"""

# Machine learning Regression
import time
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import os
try:
    os.mkdir("output")
except FileExistsError:
    pass

# global variable
date       = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
author     = 'Weineng Zhou'
ML_method  = 'Extreme Gradient Boosting Decision Tree'
abbreviate = 'xgboost'
dev_data   = './data/dev.xlsx'
val_data   = './data/validation.xlsx'

###############################################################################
# 定义机器学习方法父类(基类)
class Machine_learning:
    # 类属性(公有属性)
    def __init__(self):
        self.a = pd.read_excel('{}'.format(dev_data))
        #get independent variable
        self.X = np.array(self.a.iloc[:,0:5])
        #get the dependent variable
        y1 = self.a.iloc[:,5]*10**(-9)
        y2 = [-math.log10(x) for x in y1.tolist()]
        self.y = np.array(y2)
        #split the data at a ratio of 4:1
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=0)

ML = Machine_learning()

###############################################################################            
# 定义机器学习方法子类，并继承父类的类属性
class SVM(Machine_learning):
    # 定义调参方法
    def Tuning(self):
        self.model = SVR()
        # tune the parameters
        kernel = ['rbf']
        gamma = [1e-3, 1e-4, 1e-5]
        C = [1, 10, 100, 1000]
        # Set the parameters by 10-fold cross-validation
        self.tuned_parameters = {'kernel': kernel, 
                                 'gamma': gamma, 
                                 'C': C}
        scores = ['r2', 'neg_median_absolute_error']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(self.model, self.tuned_parameters, cv=10, scoring='%s' % score)
            start = time.time()
            clf.fit(self.X_train, self.y_train)
            print('GridSearchCV process use %.2f seconds'%(time.time()-start))
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print("Best {} found on development set:".format(score))
            print(clf.best_score_)

# 类的实例化
svm = SVM()
# 超参数优化
svm.Tuning()

###############################################################################            
class RF(Machine_learning):
    # 定义调参方法
    def Tuning(self):
        self.model = RandomForestRegressor()
        n_estimators = [200, 300, 500]
        self.tuned_parameters = {'n_estimators': n_estimators}
        scores = ['r2', 'neg_median_absolute_error']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(self.model, self.tuned_parameters, cv=10, scoring='%s' % score)
            start = time.time()
            clf.fit(self.X_train, self.y_train)
            print('GridSearchCV process use %.2f seconds'%(time.time()-start))
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print("Best {} found on development set:".format(score))
            print(clf.best_score_)
                     
# 类的实例化
rf = RF()
# 超参数优化
rf.Tuning()


###############################################################################
# 定义机器学习方法子类，并继承父类的类属性
class XGB(Machine_learning):
    # 定义调参方法
    def Tuning(self):
        self.model = XGBRegressor()
        n_estimators = [200, 300, 500]
        learning_rate = [0.01, 0.1, 1]
        max_depth = [4, 5, 6]
        self.tuned_parameters = {'n_estimators': n_estimators,
                                 'learning_rate': learning_rate,
                                 'max_depth': max_depth}
        scores = ['r2', 'neg_median_absolute_error']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(self.model, self.tuned_parameters, cv=10, scoring='%s' % score)
            start = time.time()
            clf.fit(self.X_train, self.y_train)
            print('GridSearchCV process use %.2f seconds'%(time.time()-start))
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print("Best {} found on development set:".format(score))
            print(clf.best_score_)

# 类的实例化
xgb = XGB()
# 超参数优化
xgb.Tuning()
###############################################################################

#set up the best parameters
# clf = SVR(C=1000, gamma=0.001, kernel='rbf')
# clf = RandomForestRegressor(n_estimators=300)
clf = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5)
#fit the model
model = clf.fit(ML.X_train, ML.y_train)
#predic the X_train#
y_train_pred = model.predict(ML.X_train)
#predict the X_test#
y_test_pred = model.predict(ML.X_test)
###############################################################################
# evaluate parameters
# compute the R2

model.score(ML.X_train, ML.y_train)
model.score(ML.X_test, ML.y_test)

'''
def R2(y_true, y_pred): 
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1-u/v
    return round(r2,2)
'''

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#predict the training set
ML.y_train_pred = model.predict(ML.X_train)

#predict the test set
ML.y_test_pred = model.predict(ML.X_test)

print("Statistical parameters of training set")
print("R2: %.2f" % model.score(ML.X_train,ML.y_train))
mae = mean_absolute_error(ML.y_train, ML.y_train_pred)
print("MAE: %.6f" % mae)

mse = mean_squared_error(ML.y_train, ML.y_train_pred)
print("MSE: %.6f" % mse)

rmse = (mse)**0.5
print("RMSE: %.6f" % rmse)

print("Statistical parameters of test set")
print("R2: %.2f" % model.score(ML.X_test,ML.y_test))
mae = mean_absolute_error(ML.y_test, ML.y_test_pred)
print("MAE: %.6f" % mae)

mse = mean_squared_error(ML.y_test, ML.y_test_pred)
print("MSE: %.6f" % mse)

rmse = (mse)**0.5
print("RMSE: %.6f" % rmse)

###############################################################################
# Plot feature importance
features = pd.read_excel('{}'.format(dev_data), header=0).columns
features = np.array(features.tolist()[:-1])

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center', color="#00bc57")
plt.yticks(pos, features[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig("./output/importance.tif", dpi=600)
plt.show()
###############################################################################

# scatter plot of training set
lw = 2
plt.scatter(ML.y_train,ML.y_train_pred,lw=lw, marker='.', s=50, color='#00bc57',label='Training set R2=%s' % model.score(ML.X_train,ML.y_train) )
plt.plot([ML.y_train.min(), ML.y_train.max()], [ML.y_train_pred.min(), ML.y_train_pred.max()], 
          color='black', lw=2, linestyle='-')
plt.xlabel('Experimental pIC50')
plt.ylabel('Estimated pIC50')
plt.title('eXtremely Gredient Boosting Regression')
plt.legend()
plt.savefig("./output/train_scatter.tiff", dpi=500)

###############################################################################
#scatter plot of test set
lw = 2
plt.scatter(ML.y_test,ML.y_test_pred,lw=lw, marker='.', s=50, color='#FF7F0E',label='Test set R2=%s' % model.score(ML.X_test,ML.y_test))
plt.plot([ML.y_test.min(), ML.y_test.max()], [ML.y_test_pred.min(), ML.y_test_pred.max()], 
          color='black', lw=2, linestyle='-')
plt.xlabel('Experimental pIC50')
plt.ylabel('Estimated pIC50')
plt.title('eXtremely Gredient Boosting Regression')
plt.legend()
plt.savefig("./output/test_scatter.tiff", dpi=500)

###############################################################################
# scatter plot of training set and test set
lw = 2
plt.scatter(ML.y_train,ML.y_train_pred,lw=lw, marker='.', s=50, color='#00b8e5', label='Training set R2=%s' % model.score(ML.X_train,ML.y_train))
plt.scatter(ML.y_test,ML.y_test_pred,lw=lw,  marker='.', s=50, color='#FF7F0E', label='Test set R2=%s' % model.score(ML.X_test,ML.y_test))
plt.plot([ML.y_train.min(),  ML.y_train.max()], [ML.y_train_pred.min(), ML.y_train_pred.max()], color='black', lw=2, linestyle='-')
plt.xlabel('Experimental pIC50')
plt.ylabel('Estimated pIC50')
plt.title('eXtremely Gredient Boosting Regression')
plt.legend()
plt.savefig("./output/train_test.tiff",dpi=500)

###############################################################################
# 10-fold cross validation for R2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10, shuffle=True)
R2 = cross_val_score(model, ML.X, ML.y, cv=kf)
print(R2)
print('10-fold交叉验证平均R2:', R2.mean())

###############################################################################
# output the 10-fold cross validation for R2
scores_df = pd.DataFrame(R2)
name = ['XGB']*10
name_df = pd.DataFrame(name)
M = pd.concat([name_df, scores_df], axis=1) #横向拼接数据框
M.columns=['Model', 'R2']
M.to_excel('./output/{}_R2.xlsx'.format(abbreviate))

###############################################################################
#the predicted label of validation
b = pd.read_excel('./data/validation.xlsx')
#get independent variable
X_val = np.array(b.iloc[:,0:5])

#get the dependent variable
y1 = b.iloc[:,5]*10**(-9)
y2 = [-math.log10(x) for x in y1.tolist()]
y_val = np.array(y2)

#predict the X_val
y_val_pred = model.predict(X_val)

#output the label of validation
y_val_pred_df = pd.DataFrame(y_val_pred)
y_val_pred_df.to_excel("./output/y_pred_val.xlsx")

print('the R2 of validation: %s' % model.score(X_val,y_val))

###############################################################################






