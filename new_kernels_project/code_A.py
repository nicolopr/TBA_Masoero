import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
from plotter import plot_epsilon_real, plot_epsilon_im, plot_Y0, compute_logY0

# Set parameters for a != 0
def discretize_rapidity(rapidityMax,rapidityMin,numPoints):
    rapidityVals = np.linspace(rapidityMin, rapidityMax, numPoints)
    step = (rapidityMax - rapidityMin) / (numPoints - 1)
    return rapidityVals, step

# Define phi function
def phi(beta, k):
    sqrt3 = np.sqrt(3)
    denominator = 1 + 2 * np.cosh(2 * beta)
    if k == 0:
        return -2 * sqrt3 * (2 * np.cosh(beta)) / denominator
    elif k == 1:
        return 2 * sqrt3 * np.exp(-9 / 5 * beta) / denominator
    elif k == 2:
        return 2 * sqrt3 * np.exp(-3 / 5 * beta) / denominator
    elif k == -1:
        return phi(-beta, 1)
    elif k == -2:
        return phi(-beta, 2)
    else:
        raise ValueError("Invalid k value in phi function.")

def initialize_epsilon(r, a, rapidityVals):
# Initialize epsilon values by ignoring convolution terms
    e1 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5)
    e2 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 - 1j * 2 * np.pi / 5) 
    e3 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 - 1j * 4 * np.pi / 5)
    e4 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 + 1j * 2 * np.pi / 5)
    e5 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 + 1j * 2 * np.pi / 5) 
    return e1, e2, e3, e4, e5

# Compute convolutions
def compute_convolution(coeffs, lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2):
    conv = (
        coeffs[0] * phi_0.dot(lambdas[0])
        + coeffs[1] * phi_1.dot(lambdas[1])
        + coeffs[2] * phi_2.dot(lambdas[2])
        + coeffs[3] * phi_m1.dot(lambdas[3])
        + coeffs[4] * phi_m2.dot(lambdas[4])
    )
    return (step / (2 * np.pi)) * conv / 5 #for some reason he puts in the kernels definitions an extra factor of 2*pi, and removes it here. Who knows why.

# Precompute phi matrices
def compute_phi_on_rapidities(rapidityVals):
    beta_diff = rapidityVals[:, np.newaxis] - rapidityVals[np.newaxis, :]  # Shape (numPoints, numPoints)
    phi_0 = phi(beta_diff, 0)
    phi_1 = phi(beta_diff, 1)
    phi_2 = phi(beta_diff, 2)
    phi_m1 = phi(beta_diff, -1)
    phi_m2 = phi(beta_diff, -2)
    return phi_0, phi_1, phi_2, phi_m1, phi_m2

# Iterative loop
def TBA_loop(r,a,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0
    deltaThreshold = 1e-15
    maxIterations = 5000

    #discretize rapidity space
    rapidityVals,step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi_0, phi_1, phi_2, phi_m1, phi_m2 = compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    epsilon1Old, epsilon2Old, epsilon3Old, epsilon4Old, epsilon5Old = initialize_epsilon(r, a, rapidityVals)

    while delta > deltaThreshold and iteration < maxIterations:
        iteration += 1

        # Compute convolution terms
        L1 = np.log(1 + np.exp(-epsilon1Old))
        L2 = np.log(1 + np.exp(-epsilon2Old))
        L3 = np.log(1 + np.exp(-epsilon3Old))
        L4 = np.log(1 + np.exp(-epsilon4Old))
        L5 = np.log(1 + np.exp(-epsilon5Old))

        # Compute lambda functions
        lambda1 = L1 + L2 + L3 + L4 + L5

        exp_factor = lambda n: np.exp(1j * 2 * np.pi * n / 5)
        lambda2 = L1 + exp_factor(1) * L2 + exp_factor(2) * L3 + exp_factor(-1) * L4 + exp_factor(-2) * L5
        lambda3 = L1 + exp_factor(2) * L2 + exp_factor(4) * L3 + exp_factor(-2) * L4 + exp_factor(-4) * L5
        lambda4 = L1 + exp_factor(-1) * L2 + exp_factor(-2) * L3 + exp_factor(1) * L4 + exp_factor(2) * L5
        lambda5 = L1 + exp_factor(-2) * L2 + exp_factor(-4) * L3 + exp_factor(2) * L4 + exp_factor(4) * L5

        lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5]
        coeffs_list = [
            [1, 1, 1, 1, 1],  # For conv1
            [exp_factor(0), exp_factor(1), exp_factor(2), exp_factor(-1), exp_factor(-2)],  # For conv2
            [exp_factor(0), exp_factor(2), exp_factor(4), exp_factor(-2), exp_factor(-4)],  # For conv3
            [exp_factor(0), exp_factor(-1), exp_factor(-2), exp_factor(1), exp_factor(2)],  # For conv4
            [exp_factor(0), exp_factor(-2), exp_factor(-4), exp_factor(2), exp_factor(4)],  # For conv5
        ]
        conv1 = compute_convolution(coeffs_list[0], lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2)
        conv2 = compute_convolution(coeffs_list[1], lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2)
        conv3 = compute_convolution(coeffs_list[2], lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2)
        conv4 = compute_convolution(coeffs_list[3], lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2)
        conv5 = compute_convolution(coeffs_list[4], lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2)

        # Update epsilon values
        epsilon1New = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5) - conv1
        epsilon2New = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 - 1j * 2 * np.pi / 5) - conv2
        epsilon3New = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 - 1j * 4 * np.pi / 5) - conv3
        epsilon4New = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 + 1j * 2 * np.pi / 5) - conv4
        epsilon5New = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 + 1j * 4 * np.pi / 5) - conv5

        # Compute convergence delta
        delta1 = np.max(np.abs(epsilon1New - epsilon1Old))
        delta2 = np.max(np.abs(epsilon2New - epsilon2Old))
        delta3 = np.max(np.abs(epsilon3New - epsilon3Old))
        delta4 = np.max(np.abs(epsilon4New - epsilon4Old))
        delta5 = np.max(np.abs(epsilon5New - epsilon5Old))
        delta = max(delta1, delta2, delta3, delta4, delta5)

        # Update old epsilon values
        epsilon1Old = epsilon1New
        epsilon2Old = epsilon2New
        epsilon3Old = epsilon3New
        epsilon4Old = epsilon4New
        epsilon5Old = epsilon5New

        # Print iteration info
        #print(f"Iteration {iteration}: delta = {delta}")
    print(epsilon1New[len(epsilon1New)//2],epsilon2New[len(epsilon1New)//2],epsilon3New[len(epsilon1New)//2],epsilon4New[len(epsilon1New)//2])
    return epsilon1New, epsilon2New, epsilon3New, epsilon4New, epsilon5New

# Define the function to compute the central charge 'c'
def compute_central_charge(r, a, rapidityVals, epsilon1, epsilon2, epsilon3, epsilon4, epsilon5, step):
    prefactor = (3 / np.pi**2) * r 
    scaling_factor = a / r / 5

    # Define components for each epsilon
    cosh_vals = np.cosh(rapidityVals)
    log_terms = [
        (cosh_vals + scaling_factor * np.cosh(rapidityVals / 5)) * np.log(1 + np.exp(-epsilon1)),
        (cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 - 1j * 2 * np.pi / 5)) * np.log(1 + np.exp(-epsilon2)),
        (cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 - 1j * 4 * np.pi / 5)) * np.log(1 + np.exp(-epsilon3)),
        (cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 + 1j * 2 * np.pi / 5)) * np.log(1 + np.exp(-epsilon4)),
        (cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 + 1j * 4 * np.pi / 5)) * np.log(1 + np.exp(-epsilon5)),
    ]

    # Sum all terms for the central charge calculation
    total_sum = sum([np.sum(term) * step for term in log_terms])
    central_charge = prefactor * total_sum.real

    return central_charge

def cycle_central_charge(r_min, r_max, r_step):
    central_charges=[]
    r=r_min
    while r<=r_max:
        print('valore di r:',r)
        a = A*(r**(1/5))
        e1,e2,e3,e4,e5=TBA_loop(r,a,rapidityMax,rapidityMin,numPoints)
        Y0=compute_logY0(e1,e2,e3,e4,e5)
        rapvals=discretize_rapidity(rapidityMax,rapidityMin,numPoints)
        print(Y0[len(Y0)//2])
        cc=compute_central_charge(r, a, rapidityvals, e1, e2, e3, e4, e5, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
        central_charges.append([r,cc])
        r=r+r_step
    return central_charges

#save data for epsilons
def get_filename():
    script_dir = os.path.dirname(__file__)
    return script_dir+f'/data'

def wolfram_export(data, path):
    #wolfram_expr = wl.List(*[complex(c.real, c.imag) for c in data])
    wolfram_expr = wl.List(data)

    # Specify the output file path for the .mx file
    output_file = path

    # Export the Wolfram Language expression to a .mx file
    export(wolfram_expr, output_file, target_format='wxf')


def save_charge(A, r_and_c):
    filename=get_filename()
    if not os.path.exists(filename):
        os.makedirs(filename)
    print('saved in folder'+filename)
    wolfram_export(r_and_c,f"data/ccharge_A={A}.mx")

rapidityMax = 20
rapidityMin = -20

numPoints = 1001
A = 0.01

rapidityvals=discretize_rapidity(rapidityMax,rapidityMin, numPoints)[0]

r_min=0.0001
r_max=0.001
num_of_steps = 2
r_step=(r_max-r_min)/num_of_steps

r_and_c = cycle_central_charge(r_min, r_max, r_step)
print(r_and_c)