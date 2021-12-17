#!/usr/bin/python
# -*- coding: utf-8 -*-
# from airflow import DAG
from airflow.decorators import dag,task
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn import linear_model
from train import train
import pandas as pd
from inference import inference
import os 

# def training_pipeline():
#     prediction_score = train('/', 'models/')
#     print ("+++++++++++++++++++++++++++++++++++++++++++++++++******************")
#     print (prediction_score)
#     print ("(******************************************************(((((((((((((((((((")


# def inference_pipeline():
#     my_dir = os.path.dirname(os.path.abspath(__file__))

#     production_values = inference(my_dir
#                                   + '/data/daily_dataset/block_1.csv',
#                                   my_dir + '/models/model_1.joblib')
#     print ("############################################")
#     print (production_values)
#     print ("###########################################")


# with DAG(dag_id='hello_world_dag', start_date=datetime(2021, 1, 1),
#          schedule_interval='@hourly', catchup=False) as dag:

#     task1 = PythonOperator(task_id='training_pipeline',
#                            python_callable=training_pipeline),

#     task2 = PythonOperator(task_id='inference_pipeline',
#                            python_callable=inference_pipeline)

@dag(
    dag_id='hello_world_dag', 
    start_date=datetime(2021, 1, 1),
    schedule_interval=timedelta(minutes=2), 
    catchup=False
    )
def pipelines():
    @task
    def training_pipeline():
        prediction_score = train('/', 'models/')
        print ("+++++++++++++++++++++++++++++++++++++++++++++++++******************")
        print (prediction_score)
        print ("(******************************************************(((((((((((((((((((")
    @task
    def inference_pipeline():
        my_dir = os.path.dirname(os.path.abspath(__file__))

        production_values = inference(my_dir
                                    + '/data/daily_dataset/block_1.csv',
                                    my_dir + '/models/model_1.joblib')
        print ("############################################")
        print (production_values)
        print ("###########################################")

    x = training_pipeline()
    y = inference_pipeline()

ml_dag = pipelines()
    # task1 = PythonOperator(task_id='training_pipeline',
    #                        python_callable=training_pipeline),

    # task2 = PythonOperator(task_id='inference_pipeline',
    #                        python_callable=inference_pipeline)

# task1 >> task2
