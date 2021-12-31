import pandas as pd 
import numpy as np 

"""
Input: Path to the dataset folder
OUTPUT: Returns a Pandas DataFrame containing the combined values of all households in the dataset
"""

def read_data_csv() -> pd.DataFrame:
    #read the pandas files
    energy_data = pd.read_csv("../data/daily_dataset/block_0.csv")

    # Adding 100 values of each household to the dataframe

    list_of_apps = list(energy_data["LCLid"].value_counts().index)

    dataPd = pd.DataFrame([], columns=['LCLid', 'day', 'energy_median', 'energy_mean', 'energy_max',
            'energy_count', 'energy_std', 'energy_sum', 'energy_min'])


    for apps in list_of_apps:
        filtered_data = energy_data[energy_data["LCLid"] == apps]

        if len(filtered_data) >= 100:
            temp_df = pd.DataFrame(data = filtered_data[:100], columns = filtered_data.columns)
            dataPd = dataPd.append(temp_df)
        else:
            temp_df = pd.DataFrame(data = filtered_data[:len(filtered_data)], columns= filtered_data.columns)
            dataPd = dataPd.append(temp_df)

    return dataPd