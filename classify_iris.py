#import the necessary packages
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

#construct the argument parser and parse the arguments
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-m", "--model",type=str,default="naive_bayes",help="type of python machine learning model to use")
ap.add_argument("-i","--data",type=str,default="iris",help="data to use")
args=vars(ap.parse_args())

data = {
    "iris": load_iris(),
    "cancer": load_breast_cancer()
}
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
print("[INFO] loading data '{}'....".format(args["data"]))
dataset=data[args["data"]]
(trainX,testX,trainY,testY)=train_test_split(dataset.data,dataset.target,random_state=3,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=dataset.target_names))