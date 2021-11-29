# Import libraries
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime

#Setting up MLFLOW
mlflow.set_tracking_uri('http://127.0.0.1:5000')
experiment_name = "energy_model_prediction"
mlflow.set_experiment(experiment_name)

# put the main trainingn function
def train(path: str , model_path: str) -> dict:
    training_timestamp = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
    with mlflow.start_run(run_name=f"model_{training_timestamp}"): # we can give a name to our runs.
        mlflow.autolog() 

        data_master = prepare_data()
        data = data_master.copy()
        data = data.drop(labels=["day"], axis=1) #dropping the day column
        y = data["energy_sum"]

        processed_df = preprocessing_pipeline(data)
        processed_df = processed_df.drop(labels=["energy_sum"], axis=1)
        
        X_train, X_test, y_train, y_test = perform_data_splitting(processed_df,y)

        model = training(X_train, y_train)

        y_predicted = model.predict(X_test)

        model_score = model_evaluation(model,y_predicted, y_test)

        model_path = store_model(model,model_path) # Saves the model

    return {"model_performance": model_score, "model_path": model_path}


def prepare_data():
    energy_data = pd.read_csv("../data/daily_dataset/block_0.csv")
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


def preprocessing_pipeline(data: pd.DataFrame, isInference:bool = False) -> pd.DataFrame:

    numerical_data = get_numerical_values(data) 

    categorical_data = get_categorical_columns(data)

    encoded_categorical_data = get_encoded_data(categorical_data, isInference)

    encoded_categorical_data_df = convert_to_dataframe(encoded_categorical_data)

    processed_df = concat_dataframes_on_index(numerical_data, encoded_categorical_data_df)

    return processed_df


def get_numerical_values(data: pd.DataFrame) -> pd.DataFrame:
    numerical_data = data.select_dtypes(include=[np.number])
    return numerical_data


def get_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    categorical_data = data.select_dtypes(exclude=[np.number])
    return categorical_data


def get_encoded_data(categorical_data: pd.DataFrame, isInference = False) -> np.ndarray:
    if not isInference:
        global cat_data_list
        cat_df = categorical_data
        cat_df = cat_df.dropna(axis=0)

        ohe = OneHotEncoder(handle_unknown = 'ignore')
        encoded_categorical_fit = ohe.fit(cat_df)
        encoded_categorical_data = encoded_categorical_fit.transform(cat_df)
        store_model(encoded_categorical_fit,"../models/", isEncoder=True)
        encoded_categorical_data = encoded_categorical_data.toarray()
    else:
        cat_df = categorical_data
        cat_df = cat_df.dropna(axis=0)
        path = "../models/encoder/encoder_1.joblib"
        encoded_model = load_model(path)
        encoded_categorical_data = encoded_model.transform(cat_df)
        encoded_categorical_data = encoded_categorical_data.toarray()
    return encoded_categorical_data


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


def convert_to_dataframe(data_np: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(data_np)


def concat_dataframes_on_index(numerical_data: pd.DataFrame, encoded_categorical_data: pd.DataFrame) -> pd.DataFrame:
    global num_data_list
    num_df = numerical_data
    final_df = pd.concat([num_df, encoded_categorical_data], axis=1, join="inner")
    return final_df


def perform_data_splitting(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)  
    return X_train, X_test, y_train, y_test


def training(X_train: pd.DataFrame, y_train: pd.Series):
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    model_r = regr.fit(X_train, y_train)
    model_score = regr.score(X_train,y_train)
    return model_r


def model_evaluation(model, y_pred: np.ndarray, y_test: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

prediction_score = train("/","../models/")
print(prediction_score)