

import pandas as pd 
import numpy as np 
from train import preprocessing_pipeline, prepare_data
from joblib import load
import mlflow
from datetime import datetime
import psycopg2
from config import config


def load_model(path: str) -> object:
  loaded_model = load(path)
  return loaded_model

mlflow.set_tracking_uri('http://127.0.0.1:5000')
experiment_name = "energy_model_inference_prediction"
mlflow.set_experiment(experiment_name)

# Inference pipeline
def inference(test_path: str, model_path: str) -> np.ndarray:
  training_timestamp = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
  with mlflow.start_run(run_name=f"model_{training_timestamp}"): # we can give a name to our runs.

    mlflow.autolog()
      
    data_master = prepare_data(test_path)
    data = data_master.copy()
    # data = data.drop(labels=["day"], axis=1) # dropping the day column.
    # This is the features to be sent to the database
    data_to_db = data_master.copy()

    data = data.drop(labels=["day"], axis=1) #dropping the day column
    processed_df = preprocessing_pipeline(data, isInference=True)

    processed_df = processed_df.drop(labels=["energy_sum"], axis=1)
  

    model = load_model(model_path)

    predicted_price = model.predict(processed_df)

    if len(predicted_price) != 0:
      insert_update_to_db(data_to_db, predicted_price)
    
    return predicted_price

def insert_update_to_db(data_to_db, predicted_price):
  data_to_db["energy_sum"] = predicted_price
  final_processed_data_to_db = data_to_db.to_numpy()
  query = "INSERT INTO public.predicted_data( appliance_id, day, temperaturemax, temperaturemin, pred_energy_cons) VALUES (%s, %s, %s,%s, %s)"
  try:
    params = config()
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    cur.executemany(query, final_processed_data_to_db)
    conn.commit()
    cur.close()
  except (Exception, psycopg2.DatabaseError) as error:
    print(error)


def prepare_data(filepath):
    energy_data = pd.read_csv(filepath)
    weather = pd.read_csv("../data/weather_dataset/weather_daily_darksky.csv")

    energy_data = energy_data.drop(["energy_median","energy_mean", "energy_max", "energy_count", "energy_std", "energy_min"], axis=1)
    new_weather = weather[["temperatureMaxTime","temperatureMax","temperatureMin"]]

    energy_data["day"] = pd.to_datetime(energy_data["day"])
    energy_data["day"] = energy_data["day"].dt.date # gets all the date


    new_weather["temperatureMaxTime"] = pd.to_datetime(new_weather["temperatureMaxTime"])
    new_weather["day"] = new_weather["temperatureMaxTime"].dt.date
    new_weather["day"] = new_weather["day"].sort_values()

    final_df = energy_data.merge(right=new_weather, how="inner")
    final_df = final_df.drop(["temperatureMaxTime"], axis=1)
    return final_df

production_values = inference("../data/daily_dataset/block_1.csv", "../models/model_1.joblib")
print(production_values)