import re
import nltk
from nltk.stem import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import _imaging
from sklearn.svm import SVC

from nltk.stem import PorterStemmer
file_contents = open("emailSample1.txt","r").read()
# print(file_contents)
vocabList = open("vocab.txt","r").read()
vocabList=vocabList.split("\n")[:-1]
vocabList_d={}
for ea in vocabList:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value

# print(vocabList_d)


spam_mat = loadmat("spamTrain.mat")
X_train =spam_mat["X"]
y_train = spam_mat["y"]
print(len(X_train))
print(len(X_train[0]))

# C =0.1
spam_svc = SVC(C=0.1,kernel ="linear")
print(type(spam_svc))
# exit(0)

spam_svc.fit(X_train,y_train.ravel())
print("Training Accuracy:",(spam_svc.score(X_train,y_train.ravel()))*100,"%")

spam_mat_test = loadmat("spamTest.mat")
X_test = spam_mat_test["Xtest"]
y_test =spam_mat_test["ytest"]
spam_svc.predict(X_test)
print("Test Accuracy:",(spam_svc.score(X_test,y_test.ravel()))*100,"%")

