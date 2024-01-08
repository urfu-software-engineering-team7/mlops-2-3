import os
import pandas as pd
import numpy as np
import pickle
import mlflow

from catboost import CatBoostClassifier
from sklearn import metrics
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")

cbc = pickle.load(
    open(
        "/home/modernpacifist/Documents/github-repositories/mlops-2-3/models/model.pkl",
        "rb",
    )
)
test_df = pd.read_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data_test.csv",
    sep=",",
)

X = test_df.drop("outcome", axis=1)
y = test_df["outcome"]

y_pred = cbc.predict(X)
score = metrics.accuracy_score(y, y_pred)


with mlflow.start_run():
    mlflow.log_metric("accuracy", score)
    mlflow.log_artifact(
        local_path="/home/modernpacifist/Documents/github-repositories/mlops-2-3/src/test_model.py",
        artifact_path="test_model code",
    )
    mlflow.end_run()
