import pandas as pd
import numpy as np


def data():
    data_points = pd.read_csv("Data_set_mean.csv")
    data_points_np = np.transpose(data_points.values)
    time_points = data_points_np[0, 0:29]
    mig1_n_points = data_points_np[1, 0:29]
    hxk1_points = data_points_np[2, 0:29]
    suc2_points = data_points_np[3, 0:29]
    data_conc = np.array([[mig1_n_points], [hxk1_points], [suc2_points]])
    data_conc = np.transpose(data_conc, (0, 2, 1))
    return time_points, data_conc
