import pandas as pd
import os

import mlflow
from mlflow.tracking import MlflowClient


os.environ["MLFLOW_REGISTRY_URI"] = "/home/modernpacifist/mlflow"

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("get_data")


with mlflow.start_run():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/urfu-software-engineering-team7/mlops-datasets/master/mlops-2-3/train.csv"
    )
    mlflow.log_artifact(
        local_path="/home/modernpacifist/Documents/github-repositories/mlops-2-3/src/get_data.py",
        artifact_path="get_data code",
    )
    mlflow.end_run()

data.to_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data.csv",
    sep=",",
    index=False,
)
