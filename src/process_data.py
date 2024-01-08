import os
import mlflow
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("process_data")


def work_fith_data(df, key):
    num_columns = list((df.select_dtypes(include=[int, float]).columns))
    pwt = PowerTransformer()
    num_df = pd.DataFrame(pwt.fit_transform(df[num_columns]))
    num_df.columns = num_columns

    category_columns = list((df.select_dtypes(include=[object]).columns))

    if key == 0:
        ohe = OneHotEncoder(handle_unknown="ignore")
        cat_df = pd.DataFrame(ohe.fit_transform(df[category_columns]).toarray())

    if key == 1:
        ore = OrdinalEncoder()
        cat_df = pd.DataFrame(ore.fit_transform(df[category_columns]))
        cat_df.columns = category_columns

    return num_df.join(cat_df)


train_df = pd.read_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data.csv",
    sep=",",
)
with mlflow.start_run():
    train_df = work_fith_data(train_df, 1)
    mlflow.log_artifact(
        local_path="/home/modernpacifist/Documents/github-repositories/mlops-2-3/src/process_data.py",
        artifact_path="process_data code",
    )
    mlflow.end_run()


train_df.to_csv(
    "/home/modernpacifist/Documents/github-repositories/mlops-2-3/datasets/data_processed.csv",
    sep=",",
    index=False,
)
