from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os
import mahotas as mt

from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",type=str,default="3scenes",help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model",type=str,default="knn",help="type of python machine learning model to use")
ap.add_argument("-g", "--testing",type=str,default="image_try2.jpeg",help="predict from model")
args=vars(ap.parse_args())

#Cara pake RGB Biasa
def extract_color_stats(image):
    (R,G,B)=image.split()
    fitur=[np.mean(R),np.mean(G),np.mean(B),np.std(R),np.std(G),np.std(B)]
    return fitur

#Cara pake RGB tambahan
def extract_color_stats_add(image):
    (R,G,B)=image.split()
    fitur=[np.mean(R),np.mean(G),np.mean(B),np.std(R),np.std(G),np.std(B),np.max(R),np.max(G),np.max(B),np.min(R),np.min(G),np.min(B)]
    return fitur

models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs",multi_class="auto"),
    "svm": SVC(kernel="rbf",gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

print("[INFO] extracting image features...")
imagePaths=paths.list_images(args["dataset"])
data=[]
labels=[]

label_gambar=os.listdir(args["dataset"])

for imagePath in imagePaths:
    image=Image.open(imagePath)
    fitur=extract_color_stats_add(image)
    data.append(fitur)
    label=imagePath.split(os.path.sep)[-2]
    labels.append(label)
le=LabelEncoder()
labels=le.fit_transform(labels)
label
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25)

print("[INFO] using '{}' model".format(args["model"]))
model=models[args["model"]]
model.fit(trainX,trainY)

print("[INFO] evaluating ...")
predictions=model.predict(testX)
print(classification_report(testY,predictions,target_names=le.classes_))

image=Image.open(args["testing"])
#image.show()
fitur=extract_color_stats_add(image)
predictions=model.predict([fitur])
print("prediksi gambar: ")
#int(predictions)
#if predictions==0:
#    hasil="coast"
#elif predictions==1:
#    hasil="forest"
#elif predictions==2:
#    hasil="highway"
#else:
#    hasil="apple"
print(label_gambar[int(predictions)])