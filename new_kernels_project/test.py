from TBA import TBA_loop
from plotter import plot_epsilon_real, plot_epsilon_im, plot_Y0, compute_logY0
from convolutions import discretize_rapidity
from central_charge import compute_central_charge, cycle_central_charge, cycle_central_charge2
from savedata import save_charge
import numpy as np
import os

R=1
Avals=[1]

rapidityMax = 20
rapidityMin = -20

numPoints = 1001

rapidityVals=discretize_rapidity(rapidityMax,rapidityMin,numPoints)


r_min=0.001
r_max=0.01
num_of_steps = 2
r_step=(r_max-r_min)/num_of_steps

for A in Avals:
    rvals=[0.01]
    r_and_c = cycle_central_charge2(rvals, R, A, rapidityMax,rapidityMin,numPoints)
    print(r_and_c)
    # save_charge(A,r_and_c)