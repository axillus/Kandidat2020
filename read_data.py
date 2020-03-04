import pandas as pd


def read_data():
    data_points = pd.read_csv("Data_set_mean.csv")
    print(data_points)


read_data()