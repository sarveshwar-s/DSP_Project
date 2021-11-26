# Training pipeline functions

import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cat_data_list = ["energy_mean", "Electrical"]
num_data_list = ["energy_mean", "energy_sum"]


# Loads a dataset from the given path 
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


# Gets numerical columns from the dataset
def get_numerical_values(data: pd.DataFrame) -> pd.DataFrame:
    numerical_data = data.select_dtypes(include=[np.number])
    return numerical_data


# Gets categorical columns from the dataset
def get_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    categorical_data = data.select_dtypes(exclude=[np.number])
    return categorical_data

# Performing normalization of numerical data
def min_max_normalization(numerical_data: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaled_data = MinMaxScaler(numerical_data)
    # display(scaled_data)
    return scaled_data

# Performing encoding -> Feature engineering
def get_encoded_data(categorical_data: pd.DataFrame, isInference = False) -> np.ndarray:
    if not isInference:
        global cat_data_list
        cat_df = categorical_data[cat_data_list]
        cat_df = cat_df.dropna(axis=0)

        ohe = OneHotEncoder(handle_unknown = 'error')
        encoded_categorical_fit = ohe.fit(cat_df)
        encoded_categorical_data = encoded_categorical_fit.transform(cat_df)
        store_model(encoded_categorical_fit,"../models/", isEncoder=True)
        encoded_categorical_data = encoded_categorical_data.toarray()
    else:
        global cat_data_list
        cat_df = categorical_data[cat_data_list]
        cat_df = cat_df.dropna(axis=0)
        path = "../models/encoder/encoder_1.joblib"
        encoded_model = load_model(path)
        encoded_categorical_data = encoded_model.transform(cat_df)
        encoded_categorical_data = encoded_categorical_data.toarray()
    return encoded_categorical_data

# Convertion to dataFrame
def convert_to_dataframe(data_np: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(data_np)


def concat_dataframes_on_index(numerical_data: pd.DataFrame, encoded_categorical_data: pd.DataFrame) -> pd.DataFrame:
    global num_data_list
    num_df = numerical_data[num_data_list]
    final_df = pd.concat([num_df, encoded_categorical_data], axis=1, join="inner")
    return final_df

# Doing data split 
def perform_data_splitting(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X,y[:-1],test_size=0.2,random_state=42)  
    return X_train, X_test, y_train, y_test

# Training the data
def training(X_train: pd.DataFrame, y_train: pd.Series):
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)
    return model

# Performs model prediction 
def model_prediction(model, X_test) -> np.ndarray:
    y_pred = model.predict(X_test)
    return y_pred

# Performs model evaluation 
def model_evaluation(model, y_pred: np.ndarray, y_test: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

# Saves the model
def store_model(model, model_path: str, isEncoder: bool = False) -> str:
    from joblib import dump, load
    if not isEncoder:
        model_path = model_path + "model_1.joblib"
        dump(model, model_path)
    else:
        model_encoder_path = "../models/encoder/encoder_1.joblib"
        dump(model, model_encoder_path)
        return model_encoder_path
    return model_path


def load_model(path: str):
    from joblib import dump, load
    loaded_model = load(path)
    return loaded_model


def preprocessing_pipeline(data: pd.DataFrame, isInference:bool = False) -> pd.DataFrame:

    numerical_data = get_numerical_values(data) 

    categorical_data = get_categorical_columns(data)

    encoded_categorical_data = get_encoded_data(categorical_data, isInference)

    encoded_categorical_data_df = convert_to_dataframe(encoded_categorical_data)

    processed_df = concat_dataframes_on_index(numerical_data, encoded_categorical_data_df)

    return processed_df


def train(path: str , model_path: str) -> dict:
    
    data_master = load_data(path)
    data = data_master.copy()

    y = data["SalePrice"]

    processed_df = preprocessing_pipeline(data)

    X_train, X_test, y_train, y_test = perform_data_splitting(processed_df,y)

    model = training(X_train, y_train)

    y_predicted = model_prediction(model, X_test)

    model_score = model_evaluation(model,y_predicted, y_test)

    model_path = store_model(model,model_path) # Saves the model

    return {"model_performance": model_score, "model_path": model_path}





