import pandas as pd
import numpy as np


def data():
    data_points = pd.read_csv("Data_set_mean.csv")
    data_points_np = np.transpose(data_points.values)
    time_points = data_points_np[0, :]
    mig1_n_points = data_points_np[1, :]
    hxk1_points = data_points_np[2, :]
    suc2_points = data_points_np[3, :]
    data_conc = np.transpose(np.array([[mig1_n_points], [hxk1_points], [suc2_points]]), (0, 2, 1))
    return time_points, data_conc


