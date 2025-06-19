import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
import time
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

def initialize_epsilon(r, rapidityVals):
# Initialize epsilon values by ignoring convolution terms
    e1 = r * np.cosh(rapidityVals)
    return e1

# Compute convolutions
def compute_convolution(lambdas, step, phi_0):
    conv = (
        phi_0.dot(lambdas)
    )
    return (step / (2 * np.pi)) * conv

# Precompute phi matrices
def compute_phi_on_rapidities(rapidityVals):
    beta_diff = rapidityVals[:, np.newaxis] - rapidityVals[np.newaxis, :]  # Shape (numPoints, numPoints)
    phi_0 = phi(beta_diff, 0)
    return phi_0

# Iterative loop
def TBA_loop(r,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0
    deltaThreshold = 1e-15
    maxIterations = 5000

    #discretize rapidity space
    rapidityVals,step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi_0= compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    epsilon1Old = initialize_epsilon(r, rapidityVals)

    while delta > deltaThreshold and iteration < maxIterations:
        iteration += 1

        # Compute convolution terms
        L1 = np.log(1 + np.exp(-epsilon1Old))
        # Compute lambda functions
        
        conv1 = compute_convolution(L1, step, phi_0)
        # Update epsilon values
        epsilon1New = r * np.cosh(rapidityVals) - conv1
        # Compute convergence delta
        delta = np.max(np.abs(epsilon1New - epsilon1Old))
        

        # Update old epsilon values
        epsilon1Old = epsilon1New
        # Print iteration info
        #print(f"Iteration {iteration}: delta = {delta}")
    return epsilon1New

# Define the function to compute the central charge 'c'
def compute_central_charge(r, rapidityVals, epsilon1, step):
    prefactor = (3 / np.pi**2) * r

    # Define components for each epsilon
    cosh_vals = np.cosh(rapidityVals)
    log_terms = [
        (cosh_vals) * np.log(1 + np.exp(-epsilon1)),
    ]

    # Sum all terms for the central charge calculation
    total_sum = sum([np.sum(term) * step for term in log_terms])
    central_charge = prefactor * total_sum.real

    return central_charge
def plot_epsilon_real(rapidityVals, epsilon0New):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, epsilon0New.real, label='ε0')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Re')
    plt.show()
    
def cycle_central_charge(r_min, r_max, r_step):
    central_charges=[]
    r=r_min
    while r<=r_max:
        print('valore di r:',r)
        e1=TBA_loop(r,rapidityMax,rapidityMin,numPoints)
        # plot_Y0(rapvals[0],logY0)
        # plot_epsilon_real(rapvals[0],e1,e2,e3,e4,e5)
        cc=compute_central_charge(r, rapidityvals, e1, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
        central_charges.append([r,cc])
        r=r+r_step
    return central_charges



rapidityMax = 20
rapidityMin = -20

numPoints = 10001
A = 0

rapidityvals=discretize_rapidity(rapidityMax,rapidityMin, numPoints)[0]

r_min=0.00000000001
r_max=0.0000001
num_of_steps = 2
r_step=(r_max-r_min)/num_of_steps
start = time.time()
r_and_c = cycle_central_charge(r_min, r_max, r_step)
end = time.time()
print("Python loop time: {:.4f} seconds".format(end - start))
print(r_and_c)
