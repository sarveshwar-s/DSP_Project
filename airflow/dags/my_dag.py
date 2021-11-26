from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
import datetime
from datetime import timedelta
from random import randint


# from app.inference import inference
# from app.training import training

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'start_date': datetime.datetime(2011, 12, 3),
    'end_date': datetime.datetime(2012, 6, 1),
    'email': ['airflow@airflow.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        'dsp_project',
        default_args=default_args,
        description='DSP Project',
        schedule_interval='@daily'
) as dag:
    def read_data(file_path: str):
        return 0


    def inference():
        return 0


    def training():
        return 0

    data_input = PythonOperator(
        task_id='train_model',
        python_callable=read_data
    )

    training_model = PythonOperator(
        task_id='training_model',
        python_callable=training
    )
    pred_model = PythonOperator(
        task_id='pred_model',
        python_callable=inference
    )

    data_input >> training_model >> pred_model


