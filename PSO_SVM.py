# Md. Raisul Islam Evan
# MBSTU, CSE

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from FS.pso_svm import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# load data
data  = pd.read_csv('D:/1. CE18051_Evan_MBSTU/Thesis/Last_Code/DataSet/Final.csv')
data  = data.values
feat  = np.asarray(data[:, 1:42])
label = np.asarray(data[:, 42])


# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
#k    = 5     # k-value in KNN
N    = 100    # number of particles
T    = 10   # maximum number of iterations

c1  = 2         # cognitive factor
c2  = 2         # social factor 
w   = 0.9       # inertia weight
opts = {'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = SVC()
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# Precision = TP/(TP + FP)
# Recall = TP/(TP + FN)
# F1 -score = 2 x [(Precision x Recall) / (Precision + Recall)]
# Support -> rows with a matching key in the main dataset

print(classification_report(y_valid, y_pred))

# number of selected features
num_feat = fmdl['nf']
print("No of Feature:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()

