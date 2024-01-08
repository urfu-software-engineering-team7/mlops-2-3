import os
import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_test_split")

data = pd.read_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data_processed.csv",
    sep=",",
)

test_size = 0.2
random_state = 42


with mlflow.start_run():
    X = data.drop("outcome", axis=1)
    y = data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_df = X_train.join(y_train)
    test_df = X_test.join(y_test)

    mlflow.log_param("test size", test_size)
    mlflow.log_param("random state", random_state)
    mlflow.log_artifact(
        local_path="/home/modernpacifist/Documents/github-repositories/mlops-2-3/src/train_test_split_data.py",
        artifact_path="train_test_split_data code",
    )
    mlflow.end_run()

train_df.to_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data_train.csv",
    sep=",",
    index=False,
)
test_df.to_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data_test.csv",
    sep=",",
    index=False,
)
