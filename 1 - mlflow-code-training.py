import os
import io
import time
import itertools
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest
import joblib
from pyspark.sql import functions as F
from pyspark.sql.types import *
import cml.data_v1 as cmldata

# -------------------------------
# 0) Output dirs
# -------------------------------
MODEL_DIR = "models_pkl"
CSV_DIR = "scored_csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# -------------------------------
# 1) Spark connection + data load
# -------------------------------
CONNECTION_NAME = "se-aws-edl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

EXAMPLE_SQL_QUERY = """
SELECT laptop_id, latitude, longitude, temperature, event_ts
FROM workshop.laptop_data_user001
LIMIT 500
"""

df_spark = spark.sql(EXAMPLE_SQL_QUERY)
df_spark = df_spark.withColumn("temperature", F.col("temperature").cast(DoubleType()))
df_spark = df_spark.withColumn("event_ts", F.col("event_ts").cast("string"))

df = df_spark.toPandas().dropna(subset=["latitude", "longitude", "temperature"])
X = df[["latitude", "longitude", "temperature"]].astype(float)

# -------------------------------
# 2) MLflow experiment setup
# -------------------------------
experiment_name = os.getenv("EXPERIMENT_NAME", "anomaly_detection_experiment_training")
experiment = mlflow.set_experiment(experiment_name)

# -------------------------------
# 3) Param grid + multi-run loop
# -------------------------------
param_grid = {
    "contamination": [0.05, 0.10, 0.15],
    "n_estimators": [100], #300
    "max_features": [1.0], #0.8
    "random_state": [42], #7
}
grid = list(itertools.product(
    param_grid["contamination"],
    param_grid["n_estimators"],
    param_grid["max_features"],
    param_grid["random_state"],
))

print(f"Running {len(grid)} runs in experiment '{experiment.name}' (ID={experiment.experiment_id})")

for contamination, n_estimators, max_features, seed in grid:
    run_name = f"IF_c{contamination}_n{n_estimators}_mf{max_features}_rs{seed}"
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        # Params
        mlflow.log_params({
            "model_type": "IsolationForest",
            "contamination": contamination,
            "n_estimators": n_estimators,
            "max_features": max_features,
            "random_state": seed,
        })
        mlflow.set_tags({
            "dataset.sql": EXAMPLE_SQL_QUERY.strip(),
            "dataset.rows": str(len(df)),
        })

        # Train
        t0 = time.time()
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=seed,
        )
        model.fit(X)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", float(train_time))

        # Predict
        raw_preds = model.predict(X)  # -1 anomaly, 1 normal
        anomalies = np.where(raw_preds == -1, 1, 0)

        # Metrics (unsupervised)
        anomaly_rate = float(np.mean(anomalies))
        num_anomalies = int(np.sum(anomalies))
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("num_anomalies", num_anomalies)

        # Signature + model
        try:
            signature = infer_signature(X, raw_preds)
        except Exception:
            signature = None
        mlflow.sklearn.log_model(model, "isolation_forest_model", signature=signature)

        # ----- Save artifacts to folders -----
        # 1) PKL model -> models_pkl/
        model_path = os.path.join(MODEL_DIR, f"anomaly_model_{run.info.run_id}.pkl")
        joblib.dump(model, model_path)

        # 2) Scored CSV sample -> scored_csv/
        df_scored_sample = df.copy()
        df_scored_sample["anomaly"] = anomalies
        sample = df_scored_sample.sample(n=min(200, len(df_scored_sample)), random_state=0)
        sample_path = os.path.join(CSV_DIR, f"scored_{run.info.run_id}.csv")
        sample.to_csv(sample_path, index=False)

        # Log the CSV artifact to MLflow as well (optional)
        mlflow.log_artifact(sample_path)

        # Print quick confirmation and the metrics actually logged
        client = MlflowClient()
        run_data = client.get_run(run.info.run_id).data
        print(f"[Run {run.info.run_id}] saved -> {model_path}, {sample_path} | metrics:", run_data.metrics)

print("All runs complete.")
