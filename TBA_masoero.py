import numpy as np
import matplotlib.pyplot as plt
import math
import os
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export

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
    e5 = r * np.cosh(rapidityVals) + a * np.cosh(rapidityVals / 5 + 1j * 4 * np.pi / 5)
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
    return (step / (2 * np.pi)) * conv / 5


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
    deltaThreshold = 1e-6
    maxIterations = 100

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
        print(f"Iteration {iteration}: delta = {delta}")
    return epsilon1New, epsilon2New, epsilon3New, epsilon4New, epsilon5New

# Plot the results
def plot_epsilon_real(rapidityVals, epsilon1New, epsilon2New, epsilon3New, epsilon4New, epsilon5New):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, epsilon1New.real, label='ε1')
    plt.plot(rapidityVals, epsilon2New.real, label='ε2')
    plt.plot(rapidityVals, epsilon3New.real, label='ε3')
    plt.plot(rapidityVals, epsilon4New.real, label='ε4')
    plt.plot(rapidityVals, epsilon5New.real, label='ε5')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Re')
    plt.show()
def plot_epsilon_im(rapidityVals, epsilon1New, epsilon2New, epsilon3New, epsilon4New, epsilon5New):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, epsilon1New.imag, label='ε1')
    plt.plot(rapidityVals, epsilon2New.imag, label='ε2')
    plt.plot(rapidityVals, epsilon3New.imag, label='ε3')
    plt.plot(rapidityVals, epsilon4New.imag, label='ε4')
    plt.plot(rapidityVals, epsilon5New.imag, label='ε5')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Im')
    plt.show()

#compute ln(1+Y^-1)
def compute_logY0(epsilon1New, epsilon2New, epsilon3New, epsilon4New, epsilon5New):
    epsilonfn=0.2*(epsilon1New.real+epsilon2New.real+epsilon3New.real+epsilon4New.real+epsilon5New.real)
    Yfun=[math.log(1+math.exp(-x)) for x in epsilonfn]
    return Yfun

#plot ln(1+Y^-1)
def plot_Y0(rapidityVals, Yfun):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, Yfun, label='ε1')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Re')
    plt.show()

#save data for epsilons
def get_filename(r, a):
    script_dir = os.path.dirname(__file__)
    return script_dir+f'/data/a={a}', script_dir+f'/data/a={a}/r={r}'

def wolfram_export(data, path):
    wolfram_expr = wl.List(*[complex(c.real, c.imag) for c in data])

    # Specify the output file path for the .mx file
    output_file = path

    # Export the Wolfram Language expression to a .mx file
    export(wolfram_expr, output_file, target_format='wxf')


def save_epsilon(r, a, epsilon1, epsilon2, epsilon3, epsilon4, epsilon5):
    path=get_filename(r,a)[0]
    filename=get_filename(r,a)[-1]
    if not os.path.exists(filename):
        os.makedirs(filename)
    print('saved in folder'+filename)
    wolfram_export(epsilon1,f"data/a={a}/r={r}/e1.mx")
    wolfram_export(epsilon2,f"data/a={a}/r={r}/e2.mx")
    wolfram_export(epsilon3,f"data/a={a}/r={r}/e3.mx")
    wolfram_export(epsilon4,f"data/a={a}/r={r}/e4.mx")
    wolfram_export(epsilon5,f"data/a={a}/r={r}/e5.mx")