import pandas as pd 
import numpy as np 
from app.training import preprocessing_pipeline, prepare_data



def load_model(path: str) -> object:
  from joblib import dump, load
  loaded_model = load(path)
  return loaded_model


# Inference pipeline
def inference(test_path: str, model_path: str) -> np.ndarray:

  data_master = prepare_data()
  data = data_master.copy()

  processed_df = preprocessing_pipeline(data, isInference=True)
  processed_df = processed_df.drop(labels=["energy_sum"], axis=1)

  model = load_model(model_path)

  predicted_price = model.predict(processed_df)

  return predicted_price


production_values = inference("../data/daily_dataset/block_1.csv", "../models/model_1.joblib")
print(production_values)
