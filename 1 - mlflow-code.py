import os
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
# 1. Connect to Spark via CDSW
# -------------------------------
CONNECTION_NAME = "se-aws-edl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# -------------------------------
# 2. Load data from Impala/Hive
# -------------------------------
EXAMPLE_SQL_QUERY = "SELECT laptop_id, latitude, longitude, temperature, event_ts FROM workshop.laptop_data_user001 limit 500"
df_spark = spark.sql(EXAMPLE_SQL_QUERY)

# Cast columns
df_spark = df_spark.withColumn("temperature", F.col("temperature").cast(DoubleType()))
df_spark = df_spark.withColumn("event_ts", F.col("event_ts").cast("string"))

# Convert to pandas
df = df_spark.toPandas()
df = df.dropna(subset=["latitude", "longitude", "temperature"])

# Features
X = df[["latitude", "longitude", "temperature"]]

# -------------------------------
# 3. Setup MLflow Experiment
# -------------------------------
experiment_name = os.getenv("EXPERIMENT_NAME", "anomaly_detection_experiment")
experiment = mlflow.set_experiment(experiment_name)

# -------------------------------
# 4. Train and Log Model with MLflow
# -------------------------------
contamination_rate = 0.1
random_state = 42

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log parameters
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", contamination_rate)
    mlflow.log_param("random_state", random_state)

    # Train model
    model = IsolationForest(contamination=contamination_rate, random_state=random_state)
    model.fit(X)

    # Predictions (-1 = anomaly, 1 = normal)
    df["anomaly"] = model.predict(X)

    # Convert -1 → 1 (anomaly), 1 → 0 (normal)
    df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    # Calculate and log metrics
    anomaly_rate = df["anomaly"].mean()
    mlflow.log_metric("anomaly_rate", anomaly_rate)

    # Infer model signature
    signature = infer_signature(X, model.predict(X))

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "isolation_forest_model", signature=signature)

    print("Model logged to MLflow.")

# -------------------------------
# 5. Save Model Locally (optional)
# -------------------------------
joblib.dump(model, "anomaly_model.pkl")
print("Model trained and saved as anomaly_model.pkl")

# -------------------------------
# 6. Write back results to Hive
# -------------------------------
df['laptop_id'] = pd.to_numeric(df['laptop_id'], errors='coerce').astype('Int64')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['event_ts'] = df['event_ts'].astype(str)

# Drop bad rows
df = df.dropna(subset=['laptop_id', 'latitude', 'longitude', 'temperature'])

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