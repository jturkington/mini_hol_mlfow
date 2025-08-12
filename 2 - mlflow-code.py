import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest
from pyspark.sql import functions as F
from pyspark.sql.types import *
import cml.data_v1 as cmldata
import joblib

# -------------------------------
# 0. Output directories
# -------------------------------
MODEL_DIR = "models_pkl"
CSV_DIR = "scored_csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# -------------------------------
# 1. Connect to Spark 
# -------------------------------
CONNECTION_NAME = "se-aws-edl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# -------------------------------
# 2. Load data from Iceberg (5K rows)
# -------------------------------
EXAMPLE_SQL_QUERY = """
SELECT laptop_id, latitude, longitude, temperature, event_ts
FROM workshop.laptop_data_user001
LIMIT 1000
"""
df_spark = spark.sql(EXAMPLE_SQL_QUERY)

# Cast columns
df_spark = df_spark.withColumn("temperature", F.col("temperature").cast(DoubleType()))
df_spark = df_spark.withColumn("latitude", F.col("latitude").cast(DoubleType()))
df_spark = df_spark.withColumn("longitude", F.col("longitude").cast(DoubleType()))
df_spark = df_spark.withColumn("event_ts", F.col("event_ts").cast("string"))

# Convert to pandas & clean
df = df_spark.toPandas()
df = df.dropna(subset=["latitude", "longitude", "temperature"])
# Features (as floats)
X = df[["latitude", "longitude", "temperature"]].astype(float)

# -------------------------------
# 3. Setup MLflow Experiment
# -------------------------------
experiment_name = os.getenv("EXPERIMENT_NAME", "anomaly_detection_experiment_batch")
experiment = mlflow.set_experiment(experiment_name)

# -------------------------------
# 4) Train and Log Model with MLflow
# -------------------------------

# Choose mode: "fast" for speed, "thorough" for stability/accuracy
MODE = "fast"  # or "thorough"

if MODE == "fast":
    params = {
        "model_type": "IsolationForest",
        "contamination": 0.10,   # typical default; doesn't affect speed much
        "n_estimators": 50,      # fewer trees = faster
        "max_samples": 256,      # subsample per tree = fast & scalable
        "max_features": 0.8,     # fewer features per split = faster
        "random_state": 42,
        "n_jobs": -1,            # use all cores
    }
else:  # MODE == "thorough"
    params = {
        "model_type": "IsolationForest",
        "contamination": 0.05,   # more conservative anomaly flagging
        "n_estimators": 500,     # more trees = more stable (slower)
        "max_samples": 1.0,      # use ALL samples per tree
        "max_features": 1.0,     # use ALL features per split
        "random_state": 42,
        "n_jobs": -1,            # use all cores
    }

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"IF_{MODE.upper()}") as run:
    mlflow.log_params(params)
    mlflow.set_tag("objective", MODE)

    # Train
    t0 = time.time()
    model = IsolationForest(
        contamination=params["contamination"],
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )
    model.fit(X)
    train_time = time.time() - t0
    mlflow.log_metric("train_time_sec", float(train_time))

    # Predict (-1 anomaly, 1 normal) -> {1 anomaly, 0 normal}
    preds = model.predict(X)
    df["anomaly"] = (preds == -1).astype(int)

    # Reference metrics
    anomaly_rate = float(df["anomaly"].mean())
    mlflow.log_metric("anomaly_rate", anomaly_rate)
    mlflow.log_metric("num_anomalies", int(df["anomaly"].sum()))

    # Signature + model to MLflow
    try:
        signature = infer_signature(X, preds)
    except Exception:
        signature = None
    mlflow.sklearn.log_model(model, "isolation_forest_model", signature=signature)

    # ---- Save local artifacts into folders ----
    # 1) Model PKL -> models_pkl/
    model_path = os.path.join(MODEL_DIR, f"anomaly_model_{MODE}.pkl")
    joblib.dump(model, model_path)

    # 2) Scored CSV sample -> scored_csv/
    df_scored_sample = df.copy()
    sample = df_scored_sample.sample(n=min(200, len(df_scored_sample)), random_state=0)
    csv_path = os.path.join(CSV_DIR, f"scored_{MODE}.csv")
    sample.to_csv(csv_path, index=False)

    # Optional: also log the CSV to MLflow
    mlflow.log_artifact(csv_path)

    print(
        f"[{MODE}] train_time_sec={train_time:.2f}, "
        f"anomaly_rate={anomaly_rate:.3f} | "
        f"saved model -> {model_path} | sample CSV -> {csv_path}"
    )

# -------------------------------
# 5. Write back results to Parquet Table
# -------------------------------
df["laptop_id"] = pd.to_numeric(df["laptop_id"], errors="coerce").astype("Int64")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["event_ts"] = df["event_ts"].astype(str)

# Drop bad rows
df = df.dropna(subset=["laptop_id", "latitude", "longitude", "temperature"])

# Define schema
schema = StructType([
    StructField("laptop_id", IntegerType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("event_ts", StringType(), True),
    StructField("anomaly", IntegerType(), True),
])

# Write scored data
df_results = spark.createDataFrame(df, schema=schema)
df_results.write.mode("overwrite").saveAsTable("workshop.laptop_data_scored_user001")
print("Scored data saved to workshop.laptop_data_scored_user001")
