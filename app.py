import pandas as pd
import streamlit as st
from sklearn import datasets
st.title("Data Science in Production")

st.write("""
# Explore solutions for the data of London Meters Energy Consumption
""")

dataset_name = st.sidebar.selectbox("Select Dataset",( "Nutrition","London Meters"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier",("SVM", "Classifier","Linear Regression"))

def get_dataset(dataset_name):
    if dataset_name == "London Meters":
        data = datasets.load_london_meters()
    else:
        data = datasets.load_nutrition()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X_shape)
st.write("number of classes", len(np.unique(y)))






