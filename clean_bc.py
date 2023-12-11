# Import dependencies
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

model = LogisticRegression(max_iter=1000)
model.fit(xtrain_scaled, ytrain)

# Building the predictive system
input_str = ""
while not (input_str == "0"):
    input_str = input("Enter the subject's values separated by commas or 0 to exit: ")
    if input_str == "0":
        print("Thanks for using this code")
    else:     
        inputdata = [float(value) for value in input_str.split(',')]

        # Manual scaling 
        inputdata_scaled = (inputdata - scaler.mean_) / scaler.scale_

        # Reshape and make the prediction
        inputdata_scaled_reshape = np.asarray(inputdata_scaled).reshape(1, -1)
        pred = model.predict(inputdata_scaled_reshape)

        # Display the prediction
        if pred[0] == 0:
            print("The breast cancer is malignant.")
        else:
            print("The breast cancer is benign.")
