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
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

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

np.random.seed(5)

centers=[[1,1],[-1,-1],[1,-1]]
iris=datasets.load_iris()
X=iris.data
y=iris.target

fig=plt.figure(1,figsize=(4,3))
plt.clf()
ax=Axes3D(fig, rect=[0,0,.95,1], elev=48,azim=134)

plt.cla()
pca=decomposition.PCA(n_components=3)
pca.fit(X)
X=pca.transform(X)

for name, label in [('Sentosa',0),('Versicolour',1),('Virginica',2)]:
    ax.text3D(X[y==label,0].mean(),
              X[y==label,1].mean()+1.5,
              X[y==label,2].mean(),name,
              horizontalalignment='center',
              bbox=dict(alpha=.5,edgecolor='w',facecolor='w'))
    
y=np.choose(y,[1,2,0]).astype(np.float)
ax.scatter(X[:,0],X[:,1],X[:,2],c=y,cmap=plt.cm.nipy_spectral,
           edgecolor='k')
           
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
           
plt.show()

print("[INFO] loading data '{iris}'....")
#dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(X,y,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=iris.target_names))
