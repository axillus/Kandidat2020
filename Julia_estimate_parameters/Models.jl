# Function that contains the middle-complexity ODE-model for the
# reg1 hypothesis.
function model1_reg1(du, u, h, p, t)

    # Defining the rates from parameter vector and the delay
    k1, k2, k3, k4, k5, k6, tau1 = p

    Mig1 = u[1]
    Mig1p = u[2]
    SUC2 = u[3]
    X = u[4]

    # The historic delay
    du[1] = Mig1p*k1 - Mig1*k2 + X*k6
    du[2] = -Mig1p*k1 + Mig1*k2
    du[3] = k3/(Mig1+ 0.1 ) - SUC2*k4
    du[4] = k5/(Mig1 + 0.1) - X*k6
end


# Function that contains the middle-complexity ODE-model for the
# reg1 hypothesis.
function model2_reg1(du, u, h, p, t)

    # Defining the rates from parameter vector and the delay
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau1, tau2 = p

    SNF1 = u[1]
    Mig1 = u[2]
    SUC2 = u[3]
    X = u[4]

    # The historic delay
    hist_Mig1 = h(p, t - tau1)[2]
    hist_X = h(p, t - tau2)[4]

    du[1] = k1 - k2*SNF1
    du[2] = k3 * SNF1 - k4*Mig1
    du[3] = k5 - k6*hist_Mig1^2 - k7 * hist_X
    du[4] = k8 *SUC2 - k10 * X
end
