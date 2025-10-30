import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from cleaning.main import process, conf
from ingestion.main import ingest_csv, ingest_api
from model_training.main import train_model

##################
# Ingestion Vars #
##################
csv_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Telco-Customer-Churn.csv"
api_url = "https://my.api.mockaroo.com/users"
API_HEADERS = {"X-API-Key": "2a258740"}
raw_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Raw Data"
storage_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset"

##################
# Cleaning Vars #
##################
source_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Customer Churn Data"
output_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Processed Data"

##################
# Training Vars #
##################
data_path = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Processed Data/processed_data.csv"
label_column = "Churn"
drop_columns = ["Churn", "customerID"]
model_dir = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Models/Customer Churn/models"
artifacts_dir = "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Models/Customer Churn/artifacts"

################
# Pipeline Dag #
################
with (DAG(dag_id="customer-churn-prediction-training",
          description="dags to fetch, preprocess data and train a customer churn model",
          start_date=pendulum.datetime(2025, 3, 14, tz="UTC"),
          schedule_interval="@once",
          catchup=False
          ) as dag):

    start = EmptyOperator(task_id="start")

    csv_ingestion = PythonOperator(
        python_callable=ingest_csv,
        op_args=(csv_path, raw_path),
        task_id="ingest_csv"
    )

    api_ingestion = PythonOperator(
        python_callable=ingest_api,
        op_args=(api_url, API_HEADERS, raw_path),
        task_id="ingest_api"
    )

    clean = PythonOperator(
        python_callable=process,
        op_args=(source_path, output_path, conf),
        task_id="clean"
    )

    train = PythonOperator(
        python_callable=train_model,
        op_args=(data_path, label_column, drop_columns, model_dir, artifacts_dir),
        task_id="train"
    )

    stop = EmptyOperator(task_id="stop")

    start >> [api_ingestion, csv_ingestion] >> clean >> train >> stop
