import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import csv

from read_data import data
from model_version import model, model_info, guess_k_array


def read_results():
    with open("test.csv") as csvfile:
        result_reader = csv.reader(csvfile)
        initial = 0
        for row in result_reader:
            new_result = ",".join(row)
            if new_result != "":
                if initial == 0:
                    results = np.array([(",".join(row)).split(sep=",")]).astype(np.float64)
                    print(results)
                    initial = 1
                else:
                    new_result_array = np.array(new_result.split(sep=",")).astype(np.float64)
                    results = np.vstack((results, new_result_array))
    print(results)
    print(results.shape)
    print(results[:, 0])


def test_save():
    a = np.array([[5.19754482, 1.06743208, 102.68874003, 8.03990838, 9.5829135, 0.91619954, 5.18696591, 229.38181796],
                  [1.20232305, 3.50608281, 100.53782457, 9.70836242, 11.5089053, 5.91904514, 7.06157964, 119.37536539],
                  [5.24552972, 4.36325195, 244.41280836, 8.51490433, 7.35005986, 4.56324954, 1.90956737, 1622.58204779],
                  [6.2762059, 3.95207036, 99.75676652, 17.1170465, 10.33527462, 0.71907195, 12.25365125, 133.86710417],
                  [1.96329024e+00, 4.63909600e+00, 1.06681062e+02, 2.46671124e+00, 7.45991872e+00, 3.01215629e+00,
                   4.07902318e+00, 1.62530952e+04],
                  [3.77727871, 0., 102.7532814, 8.10630224, 9.88886066, 2.67119309, 15.50106721, 1968.90730024],
                  [1.32524058, 11.85484196, 267.67329296, 14.6319316, 6.92538163, 15.86885329, 0.88934641, 68.87503556],
                  [7.62822144e+01, 1.30227426e-02, 5.56075536e-01, 3.12094030e-02, 1.52017194e+02, 1.17756944e+02,
                   2.63970429e+01, 2.70134466e+01],
                  [5.35156819e-01, 6.81281006e+01, 9.77350558e+01, 1.26133522e+00, 9.18816259e-01, 7.09651729e+01,
                   6.03359809e-01, 1.95808184e+05],
                  [6.79073960e+01, 4.65148681e+00, 2.39731868e+02, 3.62284734e+00, 2.93572758e+01, 7.02347042e+01,
                   4.39536980e-01, 1.02981266e+03],
                  [0.00000000e+00, 9.05759876e+00, 2.20487829e+02, 9.84925596e+01, 1.12078041e+02, 6.57947113e-01,
                   3.52233500e+01, 8.97948946e+02],
                  [1.36243798e-01, 5.12634721e+00, 5.16094472e+02, 5.70413887e+01, 7.25351248e+00, 4.55953012e-02,
                   2.61228775e+00, 1.71225623e+02],
                  [2.81716301e+00, 1.59677088e+02, 1.58839721e+03, 8.74250197e+01, 1.79345601e+01, 5.50484729e-02,
                   3.29306463e+00, 1.00975201e+02],
                  [19.85196268, 76.19219001, 32.98226183, 110.96488203, 12.16167114, 0., 2.08071958, 618.9107171],
                  [5.68579124, 64.95856795, 1118.32009601, 29.04162695, 94.19115486, 34.2324662, 21.40884812, 3524.99996424],
                  [3.44984298e-04, 1.36128126e+01, 8.45181628e+01, 1.96477302e+01, 2.22102913e+01, 2.20945825e+01,
                   2.37226537e+01, 2.02170770e+02],
                  [33.89484383, 11.34824381, 508.24598582, 35.56830337, 12.43585119, 7.95832341, 0.79422572, 413.07501438],
                  [0., 31.81213213, 36.5462238, 3.30802088, 191.98503012, 0.57740811, 1.84416045, 198.0390024],
                  [4.58574838, 16.78426814, 1077.41330983, 50.72481267, 640.63262101, 12.25828364, 43.19279753, 164.07237349],
                  [1.77269331e+00, 2.85027768e+01, 1.55935284e+02, 1.76439572e+02, 3.27981621e-01, 4.31266904e+01,
                   1.86414408e+01, 3.31892935e+02],
                  [5.47261397e+00, 3.36773849e+01, 7.07172138e+02, 1.61999675e+01, 7.17976257e+01, 3.13005734e+01,
                   1.34129563e-01, 1.16932704e+04]])

    with open("test.csv", "a") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        for i in range(len(a)):
            line = np.array(a[i])
            csvWriter.writerow(line)


#test_save()
read_results()
