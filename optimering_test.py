import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import time

# välj start k-värden [k_1, k_2, k_3, ... k_i] i = antal parametrar
# kör ODE med givna värden
# välj steglängd h
# kör ODE i gånger med ett k_j+h ändrat i varje fall
# använd resultatet för att beräkna gradienten
# beräkna hessianen
# se om hessianen är lösbar, om inte...fan
# ta fram p = - H^-1*grad_f
# k_ny = k+p , om abs(k - k_ny) < limit : break
# starta från början


# 3D matriser verkar inte vara långsammare än 2D, iallafall inte på sättet jag jämfört


def model_old(t, y, kinetic_constants):
    suc2 = y[0]
    hxk1 = y[1]
    mig1 = y[2]
    r1 = suc2*kinetic_constants[0]
    r2 = hxk1*kinetic_constants[1]
    r3 = hxk1*kinetic_constants[2]
    r4 = mig1*kinetic_constants[3]
    dsuc2_dt = -r1 + r2
    dhxk1_dt = r1 - r2 - r3 + r4
    dmig1_dt = r3 - r4
    dy_dt = [dsuc2_dt, dhxk1_dt, dmig1_dt]
    return dy_dt


def data_old():
    data_suc2 = np.array([1, 1.01867278, 1.03458747, 1.04816326, 1.05977747, 1.06977144, 1.0784309,  1.08590829,
                          1.09234407, 1.09788215, 1.10266992, 1.10685824, 1.11059849, 1.11395039, 1.11693021,
                          1.11957116, 1.12190974, 1.12398582, 1.12584262, 1.12752667, 1.12908718, 1.13054150,
                          1.13188430, 1.13311747, 1.13424542, 1.13527513, 1.13621611, 1.13708040, 1.13788259,
                          1.13863981, 1.13937169, 1.14008077, 1.14075587, 1.14139446, 1.14199507, 1.14255726,
                          1.14308169, 1.14357004, 1.14402506, 1.14445057, 1.14485143, 1.14523357, 1.14560397,
                          1.14597066, 1.14633429, 1.14668715, 1.14702674, 1.14735101, 1.14765845, 1.14794797,
                          1.14821903, 1.14847153, 1.14870587, 1.14892294, 1.14912412, 1.14931127, 1.14948673,
                          1.14965333, 1.14981438, 1.14997370, 1.15013557, 1.15030493, 1.15048777, 1.15067988,
                          1.15087475, 1.15106668, 1.15125072, 1.15142271, 1.15157928, 1.15171782, 1.15183651,
                          1.15193431, 1.15201094, 1.15206693, 1.15210356, 1.15212290, 1.15212780, 1.15212189,
                          1.15210957, 1.15209603, 1.15208722, 1.15208990, 1.15211157, 1.15216053, 1.15224587,
                          1.15236232, 1.15247833, 1.15258832, 1.15268878, 1.15277714, 1.15285181, 1.15291213,
                          1.15295840, 1.15299187, 1.15301475, 1.15303019, 1.15304232, 1.15305619, 1.15307783,
                          1.15311420])

    data_hxk1 = np.array([1, 0.96293548, 0.93187804, 0.90589035, 0.88412927, 0.86583257, 0.85036331, 0.8373852,
                          0.82658874, 0.81765627, 0.81026200, 0.80407197, 0.79875079, 0.79416984, 0.79029829,
                          0.78706685, 0.78439851, 0.78220846, 0.78040415, 0.77888525, 0.77754520, 0.77634929,
                          0.77531371, 0.77443813, 0.77371636, 0.77313633, 0.77268007, 0.77232376, 0.77203768,
                          0.77178626, 0.77152812, 0.77126046, 0.77101117, 0.77078840, 0.77059788, 0.77044286,
                          0.77032411, 0.77023995, 0.77018625, 0.77015638, 0.77014129, 0.77012943, 0.77010681,
                          0.77005695, 0.76998017, 0.76989535, 0.76980944, 0.76972827, 0.76965652, 0.76959777,
                          0.76955444, 0.76952785, 0.76951818, 0.76952450, 0.76954472, 0.76957566, 0.76961298,
                          0.76965123, 0.76968384, 0.76970310, 0.76970017, 0.76966473, 0.76958407, 0.76946831,
                          0.76933262, 0.76919038, 0.76905321, 0.76893099, 0.76883181, 0.76876200, 0.76872613,
                          0.76872701, 0.76876566, 0.76884137, 0.76895165, 0.76909224, 0.76925712, 0.76943850,
                          0.76962684, 0.76981082, 0.76997737, 0.77011163, 0.77019701, 0.77021513, 0.77014585,
                          0.77000125, 0.76985248, 0.76971234, 0.76958899, 0.76948840, 0.76941439, 0.76936865,
                          0.76935066, 0.76935777, 0.76938517, 0.76942588, 0.76947078, 0.76950856, 0.76952578,
                          0.76950681])

    data_mig1 = np.array([1, 1.01839174, 1.03353449, 1.04594639, 1.05609326, 1.06439599, 1.07120579, 1.0767065,
                          1.08106719, 1.08446158, 1.08706808, 1.08906978, 1.09065072, 1.09187977, 1.09277150,
                          1.09336199, 1.09369175, 1.09380571, 1.09375323, 1.09358809, 1.09336762, 1.09310921,
                          1.09280199, 1.09244440, 1.09203822, 1.09158854, 1.09110382, 1.09059585, 1.09007973,
                          1.08957393, 1.08910019, 1.08865877, 1.08823296, 1.08781713, 1.08740705, 1.08699988,
                          1.08659420, 1.08619001, 1.08578869, 1.08539305, 1.08500728, 1.08463700, 1.08428923,
                          1.08397239, 1.08368554, 1.08341750, 1.08316382, 1.08292072, 1.08268503, 1.08245426,
                          1.08222653, 1.08200062, 1.08177595, 1.08155256, 1.08133115, 1.08111307, 1.08090029,
                          1.08069544, 1.08050177, 1.08032320, 1.08016426, 1.08003034, 1.07992816, 1.07985181,
                          1.07979263, 1.07974295, 1.07969607, 1.07964630, 1.07958891, 1.07952018, 1.07943736,
                          1.07933869, 1.07922340, 1.07909170, 1.07894479, 1.07878486, 1.07861508, 1.07843961,
                          1.07826359, 1.07809315, 1.07793541, 1.07779847, 1.07769142, 1.07762434, 1.07760828,
                          1.07763643, 1.07766920, 1.07769934, 1.07772224, 1.07773446, 1.07773379, 1.07771922,
                          1.07769094, 1.07765036, 1.07760008, 1.07754392, 1.07748690, 1.07743525, 1.07739640,
                          1.07737899])
    data_conc_temp = np.array([[data_suc2], [data_hxk1], [data_mig1]])
    data_conc = np.transpose(data_conc_temp, (0, 2, 1))
    # shape = (tidsserie(3), tidssteg(100), värde(1))
    return data_conc


def model(t, y, kinetic_constants):
    suc2 = y[0]
    hxk1 = y[1]
    r1 = suc2*kinetic_constants[0]
    r2 = hxk1*kinetic_constants[1]
    dsuc2_dt = -r1 + r2
    dhxk1_dt = r1 - r2
    dy_dt = [dsuc2_dt, dhxk1_dt]
    return dy_dt


def data():
    kinetic_constants_true = [3, 6]
    t_span = [0, 1]
    y0 = [1, 1]
    t_eval = np.linspace(0, 1, num=300)
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_true), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    data_conc = np.empty([2, 300, 1])
    brus = np.random.normal(scale=0.01, size=(2, 300))
    data_conc[:, :, 0] = sol.y + brus
    return data_conc


def calc_sol_k(kinetic_constants_0, constants):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_0), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    sol_k[:, :, 0] = sol.y
    return sol_k


def calc_sol_k_step(kinetic_constants_0, constants):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    sol_k_step = np.empty([num_tidserier, num_tidsteg, num_coefficient])
    for k in range(num_coefficient):
        kinetic_constants_step = kinetic_constants_0.copy()
        kinetic_constants_step[k] = kinetic_constants_step[k] + h
        sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_step), t_span=t_span, y0=y0,
                                  method="RK45", t_eval=t_eval)
        sol_k_step[:, :, k] = sol.y
    return sol_k_step


def calc_variance(constants):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    mat_w = np.empty([num_tidserier, num_tidsteg, num_tidsteg])
    for tidserie in range(num_tidserier):
        mat_w[tidserie, :, :] = np.identity(num_tidsteg)
    return mat_w


def calc_gradient(sol_k, sol_k_step, mat_w, constants, data_concentration):
    # grad = - 2 * A_T * W * r
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    mat_r = calc_residual(sol_k, constants, data_concentration)
    diff_k_t = np.empty([num_tidserier, num_tidsteg, num_coefficient])
    mat_a = np.empty([num_tidserier, num_tidsteg, num_coefficient])  # mat_A = S_T, S = sensitivity matrix
    for k in range(num_coefficient):
        diff_k_t[:, :, k] = sol_k_step[:, :, k] - sol_k[:, :, 0]
        mat_a[:, :, k] = diff_k_t[:, :, k]/h
    mat_a_transpose = np.transpose(mat_a, (0, 2, 1))  # mat_A_T = S, S = sensitivity matrix
    mat_temp = np.matmul(mat_a_transpose, mat_w, axes=[(-2, -1), (-2, -1), (-2, -1)])
    mat_temp2 = np.matmul(mat_temp, mat_r, axes=[(-2, -1), (-2, -1), (-2, -1)])
    grad = -2 * np.add.reduce(mat_temp2, 0)
    return grad, mat_a, mat_a_transpose


def calc_residual(sol_k, constants, data_concentration):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    mat_r = np.empty([num_tidserier, num_tidsteg, 1])
    for tidsserie in range(num_tidserier):
        mat_r[tidsserie, :, 0] = data_concentration[tidsserie, :, 0] - sol_k[tidsserie, :, 0]
    return mat_r


def calc_sum_residual(sol_k, constants, data_concentration):
    mat_r = calc_residual(sol_k, constants, data_concentration)
    sum_res = np.sum(mat_r**2)
    return sum_res


def calc_approximate_hessian(mat_a, mat_a_transpose, mat_w):
    # H = 2 * A_T * W * A
    mat_temp = np.matmul(mat_a_transpose, mat_w, axes=[(-2, -1), (-2, -1), (-2, -1)])
    mat_temp2 = np.matmul(mat_temp, mat_a, axes=[(-2, -1), (-2, -1), (-2, -1)])
    hess_approx = 2 * np.add.reduce(mat_temp2, 0)
    cond_num = np.linalg.cond(hess_approx)
    print(cond_num)
    return hess_approx


def calc_approx_inverted_hessian(hess_approx):  # vet inte vilken som är bäst, ger likartade resultat
    inv_hess_approx = np.linalg.inv(hess_approx)
    # inv_hess_approx = np.linalg.pinv(hess_approx)
    return inv_hess_approx


def calc_descent_direction(grad, inv_hess_approx):
    p = - np.matmul(inv_hess_approx, grad)
    return p


def calc_step(p, kinetic_constants_0, sol_k, constants, data_concentration):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    low_sum_res = calc_sum_residual(sol_k, constants, data_concentration)
    p_transpose = np.empty(num_coefficient)
    for k in range(num_coefficient):
        p_transpose[k] = p[k]
    best_step = eps
    for new_step_length in np.linspace(eps, 1, 100):
        kinetic_constants = kinetic_constants_0 + new_step_length*p_transpose
        temp_sol_k = calc_sol_k(kinetic_constants, constants)
        temp_sum_res = calc_sum_residual(temp_sol_k, constants, data_concentration)
        if temp_sum_res < low_sum_res:
            best_step = new_step_length
        elif temp_sum_res > low_sum_res:
            break
    new_k = kinetic_constants_0 + best_step * p_transpose
    return new_k, best_step


def calc_step2(p, kinetic_constants_0, sol_k, constants, data_concentration):
    num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0 = constants
    sum_res_0 = calc_sum_residual(sol_k, constants, data_concentration)
    p_transpose = np.empty(num_coefficient)
    for k in range(num_coefficient):
        p_transpose[k] = p[k]
    best_step = np.float64(1.0)
    stop_iteration = False
    while True:
        kinetic_constants = kinetic_constants_0 + best_step * p_transpose
        temp_sol_k = calc_sol_k(kinetic_constants, constants)
        temp_sum_res = calc_sum_residual(temp_sol_k, constants, data_concentration)
        if temp_sum_res < sum_res_0:
            break
        elif temp_sum_res >= sum_res_0:
            best_step = best_step/2
        if best_step < 10**-20:
            best_step = 0
            stop_iteration = True
            break
    new_k = kinetic_constants_0 + best_step * p_transpose
    return new_k, best_step, stop_iteration


def initiation_of_variables():
    k_array = np.array([4, 7], np.float64)
    num_coefficient = len(k_array)
    num_tidserier = 2
    t_eval = np.linspace(0, 1, num=300)
    num_tidsteg = len(t_eval)
    eps = np.finfo(float).eps
    h = np.sqrt(eps)
    t_span = [0, 1]
    suc2_0 = 1
    hxk1_0 = 1
    y0 = np.array([suc2_0, hxk1_0])
    constants = (num_coefficient, num_tidserier, num_tidsteg, h, t_span, t_eval, y0)
    return k_array, constants


def iteration(k_array, constants, data_concentration):
    gogogo = True
    iteration_num = 0
    while gogogo:
        solution_k = calc_sol_k(k_array, constants)
        solution_k_step = calc_sol_k_step(k_array, constants)
        matrix_w = calc_variance(constants)
        gradient, matrix_a, matrix_a_transpose = calc_gradient(solution_k, solution_k_step, matrix_w, constants,
                                                               data_concentration)
        approximated_hessian = calc_approximate_hessian(matrix_a, matrix_a_transpose, matrix_w)
        inverted_hessian_approximation = calc_approx_inverted_hessian(approximated_hessian)
        descent_direction = calc_descent_direction(gradient, inverted_hessian_approximation)
        new_k_array, step_length, stop_iteration = calc_step2(descent_direction, k_array, solution_k, constants,
                                                              data_concentration)
        k_array = new_k_array
        sum_residue_0 = calc_sum_residual(solution_k, constants, data_concentration)
        if sum_residue_0 <= 10 ** -15:
            gogogo = False
            print("Done!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        iteration_num = iteration_num + 1
        if stop_iteration:
            print("Iteration stopped!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
            break
        if iteration_num % 1000 == 0:
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))


def main():
    k_array, constants = initiation_of_variables()
    data_concentration = data()
    iteration(k_array, constants, data_concentration)


main()

