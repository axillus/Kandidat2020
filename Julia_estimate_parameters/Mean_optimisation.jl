using NLopt
using Random
using Plots
using Printf
using Distributions
using DifferentialEquations
using DataFrames
using CSV
using Distributed


#=
import Pkg
Pkg.add("NLopt")
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("DifferentialEquations")
Pkg.add("DataFrames")
Pkg.add("CSV")
=#


include("Models.jl")
# TODO document structs
# Structs to hold parameter data
mutable struct ParameterValues
    rate_constants
    initial_values
    delays
end
struct ParameterInfo
    n_rate_constants
    n_initial_values
    min_init
    max_init
    min_rate
    max_rate
    min_delay
    max_delay
end
struct StateInfo
    states
    init_status
    map_init_opt_vector
    i_SUC2
    i_Mig1
    i_HXK1
end


# Function that will separate the mean-data into an array for the
# SUC2 data, Mig1 data and HXK1 data
# Args:
#   data_mean, the mean data set with standardised columns
# Returns:
#   SUC2_data, Mig1_data, HXK1_data
function split_data_set(data_mean)
    # Fix SUC2 data initial value, remove NaN observations
    SUC2_data = data_mean[:, [1, 4]]
    SUC2_data[1, 2] = SUC2_data[4, 2]
    SUC2_data = filter(row -> ! isnan(row.SUC2), SUC2_data)

    # Remove NAN Mig1 and HXK1 data_mean
    Mig1_data = data_mean[:, [1, 2]]
    Mig1_data = filter(row -> ! isnan(row.Mig1_n), Mig1_data)

    return SUC2_data, Mig1_data
end


# Function that will map the opt_vector (vector to optimise) to the
# model parameter object.
# Args:
#   param_model, the model parameters
#   opt_vector, the current value of the opt-vector
#   param_info, the parameter info
#   HXK1_data, the split HXK1_data
# Returns
#   param_model, an updated model parameter vector
function map_opt_vec_to_model(param_model, opt_vector, param_info, state_info)

    # Mapping rate constants
    param_model.rate_constants = opt_vector[1:param_info.n_rate_constants]

    # Mapping unknown initial initial_values, cases delay or not
    n_unknown_init = sum(state_info.init_status .== "u")
    if param_model.delays == empty
        param_model.initial_values[state_info.map_init_opt_vector] =
            opt_vector[param_info.n_rate_constants+1:end]
    else
        param_model.initial_values[state_info.map_init_opt_vector] =
            opt_vector[param_info.n_rate_constants+1:end-1]
        param_model.delays = opt_vector[end]
    end
    return param_model
end

# Function that will generate the initial random values for a model parameters
# Note that for the time-zero values the SUC2 value, Mig1 value and are
# generated from the file-values
# Args:
#   data_mean, the file containing the data-values
#   state_info, the information of the states
#   param_info, struct containing parameter information
# Returns
#   param, a struct with initial parameter values
function generate_init_values(data_mean, state_info, param_info)

    # Split the mean data
    SUC2_data, Mig1_data = split_data_set(data_mean)

    # Distributions for generating random start guesses
    dist_init = Uniform(param_info.min_init, param_info.max_init)
    dist_rates = Uniform(param_info.min_rate, param_info.max_rate)

    # Parametes to optimise. First part is rate rate_constants,
    # second unknown initial values, note special case if we have delay
    # Also add object with model parameters, not equal to parameters to optimise
    # as SUC2 and HXK1 at time zero are known
    n_unknown_init = sum(state_info.init_status .== "u")
    if param_info.max_delay == empty
        opt_vector = vcat(rand(dist_rates, param_info.n_rate_constants),
                        rand(dist_init, n_unknown_init))

        param_mod = ParameterValues(zeros(param_info.n_rate_constants),
                zeros(param_info.n_initial_values), empty)
    else
        dist_delay = Uniform(param_info.min_delay, param_info.max_delay)
        opt_vector = vcat(rand(dist_rates, param_info.n_rate_constants),
                        rand(dist_init, n_unknown_init),
                        rand(dist_delay, 1))
        param_mod = ParameterValues(zeros(param_info.n_rate_constants),
                zeros(param_info.n_initial_values), 0)
    end

    # Mapping the initial values for the model
    for i in 1:length(state_info.init_status)
        if state_info.init_status[i] == "u"
            continue
        elseif state_info.init_status[i] == "o"
            if state_info.states[i] == "SUC2"
                param_mod.initial_values[i] = SUC2_data[1, 2]
            elseif state_info.states[i] == "Mig1"
                param_mod.initial_values[i] = Mig1_data[1, 2]
            elseif state_info.states[i] == "HXK1"
                param_mod.initial_values[i] = HXK1_data[1, 2]
            end
        else # The case where a value if fixated
            param_mod.initial_values[i] = state_info.init_status[i]
        end
    end

    # Map opt-vector to param opt
    param_mod = map_opt_vec_to_model(param_mod, opt_vector, param_info,
        state_info)

    return param_mod, opt_vector
end


# Function that will solve the ODE-system for a given model.
# If it fails to solve the ode-system an error messege is
# produced, which will prompt the optimisation to exit.
# Args:
#   model, the ode-model
#   t_span, the time span for the model
#   model_param, the model parameter struct
# Returns
#   ode_solution object if success
#   "exit" if failed to solve the ode-system
function solve_dde_system(model, t_span, param_model, tau)

    # Defining the delay funciton
    h(p, t) = param_model.initial_values
    # Check whetever or not a delay should be estimated
    if param_model.delays == empty
        problem_param = vcat(param_model.rate_constants, tau)
    else
        problem_param = vcat(param_model.rate_constants, tau,
            param_model.delays)
    end
    success_symbol = Symbol("Success")
    dde_problem = DDEProblem(model, param_model.initial_values, h,
        t_span, problem_param)

    # Solve the problem
    alg = MethodOfSteps(Tsit5())
    dde_solution = solve(dde_problem, alg, verbose = false)

    if dde_solution.retcode != success_symbol
        @printf("Failed solving dde system\n")
        return "exit"
    else
        return dde_solution
    end
end


# Function that will calculate the sum of square cost value
# for a model.
# Args:
#   ode_sol, the ode solution
#   HXK1_data, SUC2_data, Mig1_data, the observed data
#   state_info, struct containing information of all states
# Returns:
#   cost, the cost function value for the current iteration
function calc_cost(dde_sol, SUC2_data, Mig1_data, state_info)

    cost = 0.0
    # SUC2
    if state_info.i_SUC2 != empty
        l_SUC2 = length(SUC2_data[:, 1])
        simulated_SUC2 = zeros(l_SUC2)
        for i in 1:l_SUC2
            t = SUC2_data[i, 1]
            simulated_SUC2[i] = dde_sol(t)[state_info.i_SUC2]
        end
        cost += sum((SUC2_data[:, 2] .- simulated_SUC2).^2)
    end
    # Mig1
    if state_info.i_Mig1 != empty
        l_Mig1 = length(Mig1_data[:, 1])
        simulated_Mig1 = zeros(l_Mig1)
        for i in 1:l_Mig1
            t = Mig1_data[i, 1]
            simulated_Mig1[i] = dde_sol(t)[state_info.i_Mig1]
        end
        cost += sum((Mig1_data[:, 2] .- simulated_Mig1).^2)
    end

    return cost
end


# Function that will plot the result for a six state model, where it is assumed
# that state 5 = SUC2 and state state 6 = HXK1 (as these will match with data)
# Args:
#   ode_sol, the ode solution
#   SUC2_data, the SUC2-data
#   HXK1-data, the HXK1 data
#   state_info, information of the state-names
function plot_four_state_model(param_mod, state_info, path_data, model, tau)

    # Derive general parameters
    data_mean = CSV.read(path_data)
    SUC2_data, Mig1_data = split_data_set(data_mean)
    t0 = convert(Float64, SUC2_data[1, 1])
    t_end = convert(Float64, SUC2_data[end, 1])
    time_span = (t0, t_end)

    # Solve the ODE-system
    dde_sol = solve_dde_system(model, time_span, param_mod, tau)

    # SUC2 data
    i_SUC2 = state_info.i_SUC2
    t = [dde_sol.t, SUC2_data[:, 1]]
    y = [dde_sol[i_SUC2, 1:end], SUC2_data[:, 2]]
    p1 = plot(t, y, legend=false, title=state_info.states[i_SUC2])
    # Mig1
    i_Mig1 = state_info.i_Mig1
    t = [dde_sol.t, Mig1_data[:, 1]]
    y = [dde_sol[i_Mig1, 1:end], Mig1_data[:, 2]]
    p2 = plot(t, y, legend=false, title=state_info.states[i_Mig1])
    p3 = plot(dde_sol.t, dde_sol[1, 1:end], legend=false, title=state_info.states[1])
    p4 = plot(dde_sol.t, dde_sol[4, 1:end], legend=false, title=state_info.states[4])

    plot(p1, p2, p3, p4, layout=(2, 2))
end


function target_function(opt_vector, grad, param_info, state_info,
    SUC2_data, Mig1_data, param_model, time_span, tau, model)

    if length(grad) > 0
        @printf("Cannot calculate gradient\n")
    end

    param_model = map_opt_vec_to_model(param_model, opt_vector, param_info,
        state_info)
    dde_sol = solve_dde_system(model, time_span, param_model, tau)
    if dde_sol == "exit"
        return "exit"
    end
    cost = calc_cost(dde_sol, SUC2_data, Mig1_data, state_info)
    return cost
end


# Function that will try to fit the mean data for a ODE-model provided
# by the user.
# Args:
#   path_data, path to the data set
#   param_info, a struct with information of the parameters
#   states, a vector with the states in the model
#   ode_model, the ode-model in question
#   times_run, the number of times to generate random start-guesses
# Returns
#   best_param, the best parameter vector
function test_model(path_data, param_info, state_info, model, times_run, tau)

    # Derive general parameters
    data_mean = CSV.read(path_data)
    SUC2_data, Mig1_data = split_data_set(data_mean)
    t0 = convert(Float64, SUC2_data[1, 1])
    t_end = convert(Float64, SUC2_data[end, 1])
    time_span = (t0, t_end)

    # Allocate memory for running in parallel
    n_unknown_init = convert(Int64, sum(state_info.init_status .== "u"))
    n_param = param_info.n_rate_constants + n_unknown_init
    if param_info.max_delay == empty
        opt_mat = zeros(Float64, times_run, n_param)
    else
        opt_mat = zeros(Float64, times_run, n_param+1)
    end
    func_val = zeros(times_run)

    # Minimise the function of interest
    best_cost = Inf
    #Threads.@threads for i in 1:times_run
    for i in 1:times_run
        @printf("Iteration = %d\n", i)
        # Generate start guesses
        param_model, opt_vector = generate_init_values(data_mean, state_info, param_info)
        n_param = length(opt_vector)

        # Solve the problem
        opt = Opt(:LN_SBPLX, n_param)
        opt.lower_bounds = 0
        opt.upper_bounds = Inf
        opt.maxeval = 1000

        min_objective!(opt, (opt_vector, grad) -> target_function(opt_vector, grad,
            param_info, state_info, SUC2_data, Mig1_data,
            param_model, time_span, tau, model))
        (minf, minx, ret) = optimize(opt, opt_vector)

        opt_mat[i, :] = minx
        func_val[i] = minf
    end

    best_run = argmin(func_val)
    best_opt = opt_mat[best_run, :]
    param_model, opt_vector = generate_init_values(data_mean, state_info, param_info)
    param_best = map_opt_vec_to_model(param_model, best_opt, param_info, state_info)

    return param_best
end


# Function that given a string of the states will create a StateInfo struct
# Args:
#   state_info, list with all the states
# Returns:
#   states, object with information of the states
function produce_state_info(states, observed, init_status)
    # Define the indices for SUC2, Mig1 and HXK1
    if "SUC2" in observed
        i_SUC2 = findall(x->x=="SUC2", states)[1]
    else
        i_SUC2 = empty
    end
    if "Mig1" in observed
        i_Mig1 = findall(x->x=="Mig1", states)[1]
    else
        i_Mig1 = empty
    end
    if "HXK1" in observed
        i_HXK1 = findall(x->x=="HXK1", states)[1]
    else
        i_HXK1 = empty
    end

    # For mapping from opt-vector to init vector
    map_init_opt_vector = findall(x->x=="u", init_status)

    state_info = StateInfo(states, init_status, map_init_opt_vector,
        i_SUC2, i_Mig1, i_HXK1)

    return state_info
end

function main()

    Random.seed!(52356)

    # General four states model parameters
    tau = 0
    path_data = "../Intermediate/Data_files/Data_set_mean.csv"
    states = ["SNF1", "Mig1", "SUC2", "X"]
    state_info = produce_state_info(states, ["SUC2"], [1, 1, "o", 0]) # Map initial values

    # Model 1 without a delay
    model = model1_reg1
    param_info = ParameterInfo(10, 4, 1, 4, 0, 10, empty, empty)
    best_param = test_model(path_data, param_info, state_info, model, 2, tau)
    println(best_param)

end

main()
