import numpy as np


results = np.array([1, 2, 3, 4, 5, 6])
f = open("opt_res_test.txt", "a")
f.write(str(results) + " test \n")
f.close()