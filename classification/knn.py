# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:45:56 2020

@author: weineng.zhou
"""


import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from scipy import interp
from sklearn.model_selection import StratifiedKFold

import os
try:
    os.mkdir("output")
except FileExistsError:
    pass

# global variable
date       = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
author     = 'Weineng Zhou'
ML_method  = 'K Neighbors'
abbreviate = 'knn'
dev_data   = './data/drug_molecule.xlsx'
val_data   = './data/validation.xlsx'

class_names = ['非有效药物', '有效药物']

t1 = datetime.datetime.now()
print('开始时间:', t1.strftime('%Y-%m-%d %H:%M:%S'))

###############################################################################
# load the data
a = pd.read_excel('{}'.format(dev_data))
a = np.array(a)
#get independent variable
X = a[:,:-1]
#get the dependent variable
y = a[:,-1]
#split the data at a ratio of 4:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
                                                    random_state=0)

# y_train = np.ravel(y_train)

# default model
model = KNeighborsClassifier()

# tune the parameters
weights = ['uniform', 'distance']
n_neighbors = list(range(1,20,1))
tuned_parameters = {'weights': weights, 
                  'n_neighbors': n_neighbors}
        
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    classifier = GridSearchCV(model, tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    classifier.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(classifier.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, classifier.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# set up the best parameters
classifier = KNeighborsClassifier(n_neighbors=6, weights='distance')

# fit the model
classifier.fit(X_train, y_train)
# predic by X_train
y_train_pred = classifier.predict(X_train)
# predict by X_test
y_test_pred = classifier.predict(X_test)

#the label of confusion matrix
class_names = np.array(class_names)
# plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()

# calculate the training set confusion matrix
cnf_matrix = confusion_matrix(y_train, y_train_pred)
np.set_printoptions(precision=2)
# without normalization confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='训练集')
plt.savefig('./output/train_matrix_{}.tiff'.format(abbreviate), dpi=500)
plt.show()

# calculate the test set confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)
# without normalization confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='测试集')
plt.savefig('./output/test_matrix_{}.tiff'.format(abbreviate), dpi=500)
plt.show()
###############################################################################

# 10-fold cross validation for accuracy
from sklearn.model_selection import cross_val_score
import seaborn as sns
scores = cross_val_score(model, X, y, cv=10)
scores
scores.mean()
print(scores)
print(scores.mean())

scores_df = pd.DataFrame(scores)
name = ['LGBM']*10
name_df = pd.DataFrame(name)
M = pd.concat([name_df, scores_df], axis=1) #横向拼接数据框
M.columns=['Model', 'Accuracy']
M.to_excel('./output/{}_Accuracy.xlsx'.format(abbreviate), index=False)
sns.boxplot(data=M, x = 'Model', y = 'Accuracy', color='#00b8e5')
plt.savefig("./output/boxplot.tiff", dpi=600)


###############################################################################
# the predicted label of validation
b = pd.read_excel('{}'.format(val_data))
#get independent variable
b = np.array(b)
#get independent variable
X_val = b[:,:-1]
#get the dependent variable
y_val = b[:,-1]
#predict the X_val
y_val_pred = classifier.predict(X_val)
#output the label of validation
y_val_pred_df = pd.DataFrame(y_val_pred)
y_val_pred_df.to_excel('./output/label_{}.xlsx'.format(abbreviate))

###############################################################################
# plot the confusion matrix
#calculate the validation set confusion matrix
cnf_matrix = confusion_matrix(y_val, y_val_pred)
np.set_printoptions(precision=2)

#without normalization confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='验证集')
plt.savefig('./output/val_matrix_{}.tiff'.format(abbreviate), dpi=500)


###############################################################################
ax = plt.gca()
lgbc_disp = plot_roc_curve(classifier, X_train, y_train, ax=ax, color='#00bc57', lw=2, alpha=0.8)
#ROC plot of training set
ax.plot([0, 1], [0, 1], color='#00bc57', lw=2, linestyle='--')
# ax.xlim([0.0, 1.0])
# ax.ylim([0.0, 1.05])
ax.set(xlim=[-0.05, 1.05], 
       ylim=[-0.05, 1.05], 
       xlabel='False Positive Rate (1-Specificity)',
       ylabel='Ture Positive Rate (Sensitivity)',
       title="ROC curve for training set")
ax.legend(loc="lower right")
ax.legend(loc="lower right")
plt.savefig('./output/roc_train_{}.tiff'.format(abbreviate), dpi=500)
plt.show()


ax = plt.gca()
lgbc_disp = plot_roc_curve(classifier, X_test, y_test, ax=ax, color='#00b8e5', lw=2, alpha=0.8)
#ROC plot of training set
ax.plot([0, 1], [0, 1], color='#00bc57', lw=2, linestyle='--')
# ax.xlim([0.0, 1.0])
# ax.ylim([0.0, 1.05])
ax.set(xlim=[-0.05, 1.05], 
       ylim=[-0.05, 1.05], 
       xlabel='False Positive Rate (1-Specificity)',
       ylabel='Ture Positive Rate (Sensitivity)',
       title="ROC curve for test set")
ax.legend(loc="lower right")
plt.savefig('./output/roc_test_{}.tiff'.format(abbreviate), dpi=500)
plt.show()


ax = plt.gca()
lgbc_disp = plot_roc_curve(classifier, X_val, y_val, ax=ax, color='#ff7f0e', lw=2, alpha=0.8)
#ROC plot of training set
ax.plot([0, 1], [0, 1], color='#00bc57', lw=2, linestyle='--')
# ax.xlim([0.0, 1.0])
# ax.ylim([0.0, 1.05])
ax.set(xlim=[-0.05, 1.05], 
       ylim=[-0.05, 1.05], 
       xlabel='False Positive Rate (1-Specificity)',
       ylabel='Ture Positive Rate (Sensitivity)',
       title="ROC curve for validation set")
ax.legend(loc="lower right")
plt.savefig('./output/roc_val_{}.tiff'.format(abbreviate), dpi=500)
plt.show()


###############################################################################
# 10-fold cross validation for ROC
#load the data
a = pd.read_excel('{}'.format(dev_data))
a = np.array(a)
#get independent variable
X = a[:,:-1]
#get the dependent variable
y = a[:,-1]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis
cv = StratifiedKFold(n_splits=10)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05],
       ylim=[-0.05, 1.05],
       xlabel='False Positive Rate (1-Specificity)',
       ylabel='Ture Positive Rate (Sensitivity)',
       title='{}'.format(ML_method))
ax.legend(loc='lower right')
plt.savefig('./output/roc_crossval_{}.tiff'.format(abbreviate), dpi=500)
plt.show()

###############################################################################
# 开始时间
print('开始时间:', t1.strftime('%Y-%m-%d %H:%M:%S'))
# 结束时间
t2 = datetime.datetime.now()
print('结束时间:', t2.strftime('%Y-%m-%d %H:%M:%S'))
delta = t2 - t1

if delta.seconds > 3600:
    if t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] < t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：'
              + str(int(round(delta.seconds / 3600, 0))) + '时'
              + str(int(round(delta.seconds / 60, 0) % 60)) + '分'
              + str(delta.seconds % 60) + '秒')
    elif t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] == t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：'
              + str(int(round(delta.seconds / 3600, 0))) + '时'
              + str(int(round(delta.seconds / 60, 0) % 60)) + '分'
              + '0秒')
    elif t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] > t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：'
              + str(int(round(delta.seconds / 3600, 0))) + '时'
              + str(int(round(delta.seconds / 60, 0) % 60)-1) + '分'
              + str(delta.seconds % 60) + '秒')
        
elif delta.seconds > 60:
    if t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] < t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：' + str(int(round(delta.seconds / 60, 0))) + '分'
              + str(delta.seconds % 60 +1) + '秒')
    elif t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] == t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：' + str(int(round(delta.seconds / 60, 0))) + '分'
              + '0秒')
    elif t1.strftime('%Y-%m-%d %H:%M:%S')[-2:] > t2.strftime('%Y-%m-%d %H:%M:%S')[-2:]:
        print('总共耗时：' + str(int(round(delta.seconds / 60, 0))-1) + '分'
              + str(delta.seconds % 60 +1) + '秒')

else:
    print('总共耗时：' + str(delta.seconds) + '秒')
