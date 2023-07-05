import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

import os
import numpy as np

from data_structures import support_files,directories

df = pd.read_csv(os.path.join(directories["support_files"],support_files["pose"]["name"]))

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression()), # Logistic regression
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()), # Ridge classifier
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()), # Random forest
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()), # Gradient boosting
}

# Model training
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# Model scores
max_score = ["",0]
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    score = accuracy_score(y_test, yhat)
    print(algo, score)
    if score > max_score[1]:
        max_score = [algo,score]

# Save the winning model
with open(os.path.join(directories["support_files"],'pose.pkl'), 'wb') as f:
    pickle.dump(fit_models[algo], f)