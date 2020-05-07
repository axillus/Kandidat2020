import numpy as np
import csv


def model(modell_type, t, y, kinetic_constants):
    if modell_type == "1":
        return model_1(t, y, kinetic_constants)
    elif modell_type == "1_k6_k7":
        return model_1_k6_k7(t, y, kinetic_constants)
    elif modell_type == "1_k5_k3":
        return model_1_k5_k3(t, y, kinetic_constants)
    elif modell_type == "1_k7_noll":
        return model_1_k7_noll(t, y, kinetic_constants)
    else:
        print("Not a modell!")


def model_1(t, y, kinetic_constants):
    suc2 = y[0]
    mig1 = y[1]
    mig1_phos = y[2]
    X = y[3]
    r1 = kinetic_constants[0] * mig1_phos
    r2 = mig1 * kinetic_constants[1]
    r3 = kinetic_constants[2]/(mig1 + 0.1)
    r4 = suc2*kinetic_constants[3]
    r5 = kinetic_constants[4]/(mig1 + 0.1)
    r6 = X * kinetic_constants[5]
    r7 = mig1 * kinetic_constants[6]
    dmig1_dt = r1 - r2 + r6 - r7
    dmig_phos_dt = - r1 + r2
    dsuc2_dt = r3 - r4
    dX_dt = r5 - r6
    dy_dt = [dsuc2_dt, dmig1_dt, dmig_phos_dt, dX_dt]
    return dy_dt


def model_1_k6_k7(t, y, kinetic_constants):
    suc2 = y[0]
    mig1 = y[1]
    mig1_phos = y[2]
    X = y[3]
    r1 = kinetic_constants[0] * mig1_phos
    r2 = mig1 * kinetic_constants[1]
    r3 = kinetic_constants[2]/(mig1 + 0.1)
    r4 = suc2*kinetic_constants[3]
    r5 = kinetic_constants[4]/(mig1 + 0.1)
    r6 = X * kinetic_constants[5]
    r7 = mig1 * kinetic_constants[5]
    dmig1_dt = r1 - r2 + r6 - r7
    dmig_phos_dt = - r1 + r2
    dsuc2_dt = r3 - r4
    dX_dt = r5 - r6
    dy_dt = [dsuc2_dt, dmig1_dt, dmig_phos_dt, dX_dt]
    return dy_dt


def model_1_k5_k3(t, y, kinetic_constants):
    suc2 = y[0]
    mig1 = y[1]
    mig1_phos = y[2]
    X = y[3]
    r1 = kinetic_constants[0] * mig1_phos
    r2 = mig1 * kinetic_constants[1]
    r3 = kinetic_constants[2]/(mig1 + 0.1)
    r4 = suc2*kinetic_constants[3]
    r5 = kinetic_constants[2]/(mig1 + 0.1)
    r6 = X * kinetic_constants[4]
    r7 = mig1 * kinetic_constants[5]
    dmig1_dt = r1 - r2 + r6 - r7
    dmig_phos_dt = - r1 + r2
    dsuc2_dt = r3 - r4
    dX_dt = r5 - r6
    dy_dt = [dsuc2_dt, dmig1_dt, dmig_phos_dt, dX_dt]
    return dy_dt


def model_1_k7_noll(t, y, kinetic_constants):
    suc2 = y[0]
    mig1 = y[1]
    mig1_phos = y[2]
    X = y[3]
    r1 = kinetic_constants[0] * mig1_phos
    r2 = mig1 * kinetic_constants[1]
    r3 = kinetic_constants[2]/(mig1 + 0.1)
    r4 = suc2*kinetic_constants[3]
    r5 = kinetic_constants[4]/(mig1 + 0.1)
    r6 = X * kinetic_constants[5]
    dmig1_dt = r1 - r2 + r6
    dmig_phos_dt = - r1 + r2
    dsuc2_dt = r3 - r4
    dX_dt = r5 - r6
    dy_dt = [dsuc2_dt, dmig1_dt, dmig_phos_dt, dX_dt]
    return dy_dt


def num_coeff():
    num_coefficient = 7
    return num_coefficient


def guess_k_array(runda):
    num_coefficient = num_coeff()
    k_array = np.array([10, 10, 10, 10, 10, 1], np.float64)
    vary = True
    mix_up = True
    test_specific_values = False
    if test_specific_values:
        # set your values
        k_array = np.array([43.6592032, 11.49582613, 82.95251143, 11.89430632, 12.09599779, 1.79678015], np.float64)
    else:
        if vary:
            variation = np.random.normal(scale=1, size=num_coefficient)
            k_array = k_array + variation
            k_array = np.abs(k_array)
        if mix_up:
            rand_val_coeff = np.random.randint(3, size=num_coeff())
            for i in range(num_coefficient):
                if rand_val_coeff[i] == 0:
                    k_array[i] = k_array[i] / 10
                elif rand_val_coeff[i] == 1:
                    k_array[i] = k_array[i]
                else:
                    k_array[i] = k_array[i] * 10
    return k_array


def model_info(time_points):
    vald_modell = "1"
    num_coefficient = num_coeff()
    num_tidserier = 4
    t_eval = time_points
    num_tidsteg = len(t_eval)
    eps = np.finfo(float).eps
    h = np.sqrt(eps)
    t_span = [t_eval[0], t_eval[-1]]
    suc2_0 = 4.043530
    mig1_0 = 2.649860
    mig1_phos_0 = 0
    X_0 = 0
    y0 = np.array([suc2_0, mig1_0, mig1_phos_0, X_0])
    compare_to_data = ["2", "0", False, False]
    num_compare = 2
    constants = (vald_modell, num_coefficient, num_tidserier, num_tidsteg, h)
    ode_info = (t_span, t_eval, y0)
    data_info = (compare_to_data, num_compare)
    return constants, ode_info, data_info
