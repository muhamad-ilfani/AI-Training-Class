from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
import numpy as np
import argparse

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model",type=str,default="naive_bayes",help="type of python machine learning model to use")
args=vars(ap.parse_args())

models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs",multi_class="auto"),
    "svm": SVC(kernel="rbf",gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier(),
    "pr": Perceptron(tol=1e-3,random_state=0)
}

iris = datasets.load_iris()

X=iris.data
y=iris.target
X_ori=X
X_2=X[:,0:2]

target_names=iris.target_names

pca=PCA(n_components=2)
X_r=pca.fit(X).transform(X)

lda=LinearDiscriminantAnalysis(n_components=2)
X_r2=lda.fit(X,y).transform(X)

fa=FactorAnalysis(n_components=2)
X_r3=fa.fit_transform(iris.data)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors=['navy','turquoise','darkorange']
lw=2

for color,i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_r[y==i,0],X_r[y==i,1],color=color,alpha=.8,lw=lw,
                label=target_name)
plt.legend(loc='best',shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color,i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_r2[y==i,0],X_r2[y==i,1],color=color,alpha=.8,
                label=target_name)
plt.legend(loc='best',shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.figure()
for color,i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_r3[y==i,0],X_r3[y==i,1],color=color,alpha=.8,lw=lw,
                label=target_name)
plt.legend(loc='best',shadow=False, scatterpoints=1)
plt.title('FA of IRIS dataset')

plt.figure()
for color,i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_ori[y==i,0],X_ori[y==i,1],color=color,alpha=.8,
                label=target_name)
plt.legend(loc='best',shadow=False, scatterpoints=1)
plt.title('Original of IRIS dataset')

plt.show()

print("[INFO] loading data 'iris'.... metode PCA")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X_r,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))

print("[INFO] loading data 'iris'.... metode LDA")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X_r2,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))

print("[INFO] loading data 'iris'.... metode FA")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X_r3,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))

print("[INFO] loading data 'iris'.... Original Data 4 Features")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X_ori,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))

print("[INFO] loading data 'iris'.... Original Data 2 Features")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X_2,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))