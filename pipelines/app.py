from flask import Flask, jsonify
from joblib import load
from inference import inference

app = Flask(__name__)
if __name__ == "__main__":
    app.run()

#loading the model in the api
model = load("../models/model_1.joblib")


@app.route("/")
def homepage():
    # This should route to streamlit
    model.predict([[""]])
    return "Hello Again from flask!!"

@app.route("/energy/predict/")
def prediction_api():
    prediction_list = inference("../data/daily_dataset/block_1.csv", "../models/model_1.joblib")
    values = prediction_list.tolist()
    return jsonify({"Predicted_consumption":values})