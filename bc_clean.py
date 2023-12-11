# Import dependencies
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and processing
bcdata = load_breast_cancer()

# Load data to dataframe
df = pd.DataFrame(bcdata.data, columns=bcdata.feature_names)
df["label"] = bcdata.target

# Features and target
x = df.drop(columns="label", axis=1)
y = df["label"]

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2)

# Model training
model = LogisticRegression()
model.fit(xtrain, ytrain)

# Building the predictive system
input_str = input("Enter the subject's values separated by commas: ")
inputdata = [float(value) for value in input_str.split(',')]

# Reshape and make the prediction
inputdata_np = np.asarray(inputdata)
inputdata_reshape = inputdata_np.reshape(1, -1)
pred = model.predict(inputdata_reshape)

# Display the prediction
if pred[0] == 0:
    print("The breast cancer is malignant.")
else:
    print("The breast cancer is benign.")

