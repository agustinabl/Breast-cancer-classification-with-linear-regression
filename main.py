### Import dependencies ###
import numpy as np
import pandas as pd 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

##### Data collection and processing #####
#Data collection 
bcdata=sklearn.datasets.load_breast_cancer()
print(bcdata)

#load data to dataframe and analysis
df=pd.DataFrame(bcdata.data, columns=bcdata.feature_names)
df.head()
df["label"]=bcdata.target
df.tail
df.info()
df.isnull().sum()

##### Statistics #####
#statistical measures 
df.describe()
df["label"].value_counts()
#1 is benign and 0 is malignant 
df.groupby("label").mean()

##### Feat andtarget ######
x=df.drop(columns="label", axis=1)
y= df["label"]
print(x)
print(y)

###### Split data into training and test ######
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.2, random_state=2)
print(x.shape, xtrain.shape, xtest.shape)


####### Model training #########
#Logistic regression 
model=LogisticRegression()
model.fit(xtrain, ytrain)

#accuracy score
xtrain_pred = model.predict(xtrain)
trainingdata_acc = accuracy_score(ytrain, xtrain_pred)
print("Accuracy on training data is", trainingdata_acc)

#accuracy on test data 
xtest_pred = model.predict(xtest)
testdata_acc = accuracy_score(ytest, xtest_pred)
print("Accuracy on test data is", testdata_acc)

######Building the predicive system ########
input_str = input("Enter the subject's values separated by commas: ")
inputdata = [float(value) for value in input_str.split(',')]

#reshape
inputdata_np = np.asarray(inputdata)
inputdata_reshape = inputdata_np.reshape(1, -1)

#make the prediction
pred = model.predict(inputdata_reshape)
if pred[0] == 0:
    print("The breast cancer is malignant")
else:
    print("The breast cancer is benign")