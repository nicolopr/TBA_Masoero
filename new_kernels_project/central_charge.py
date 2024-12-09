import numpy as np
from TBA import TBA_loop
from convolutions import discretize_rapidity
from plotter import compute_logY0, plot_Y0, plot_epsilon_real
# # Define the function to compute the central charge 'c'

def compute_central_charge(R, m, A, rapidityVals, epsilonm2, epsilonm1, epsilon0, epsilon1, epsilon2, step):
    prefactor = (3 / np.pi**2) 
    scaling_factor = A * R* m**(1/5) / 5

    # Define components for each epsilon
    cosh_vals = np.cosh(rapidityVals)
    log_terms = [
        (R*m*cosh_vals + scaling_factor * np.cosh(rapidityVals / 5)) * np.log(1 + np.exp(-epsilon0)),
        (R*m*cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 - 1j * 2 * np.pi / 5)) * np.log(1 + np.exp(-epsilon1)),
        (R*m*cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 - 1j * 4 * np.pi / 5)) * np.log(1 + np.exp(-epsilon2)),
        (R*m*cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 + 1j * 2 * np.pi / 5)) * np.log(1 + np.exp(-epsilonm1)),
        (R*m*cosh_vals + scaling_factor * np.cosh(rapidityVals / 5 + 1j * 4 * np.pi / 5)) * np.log(1 + np.exp(-epsilonm2)),
    ]

    # Sum all terms for the central charge calculation
    total_sum = sum([np.sum(term) * step for term in log_terms])
    central_charge = prefactor * total_sum.real

    return central_charge

def cycle_central_charge(r_min, r_max, r_step, R, A, rapidityMax,rapidityMin,numPoints):
    central_charges=[]
    m=r_min
    while m<=r_max:
        print('valore di m:',m)
        e1,e2,e3,e4,e5=TBA_loop(R, m, A,rapidityMax,rapidityMin,numPoints)
        logY0=compute_logY0(e1,e2,e3,e4,e5)
        rapvals=discretize_rapidity(rapidityMax,rapidityMin,numPoints)
        cc=compute_central_charge(R, m, A, rapvals[0], e1, e2, e3, e4, e5, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
        central_charges.append([m,cc])
        m+=r_step
    return central_charges