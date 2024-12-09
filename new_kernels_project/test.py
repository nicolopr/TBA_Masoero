from TBA import TBA_loop
from plotter import plot_epsilon_real, plot_epsilon_im, plot_Y0, compute_logY0
from convolutions import discretize_rapidity
from central_charge import compute_central_charge, cycle_central_charge
from savedata import save_charge
import numpy as np
import os

R=1
Avals=[0.01,0.1,1,10]

rapidityMax = 70
rapidityMin = -70

numPoints = 1001

rapidityVals=discretize_rapidity(rapidityMax,rapidityMin,numPoints)


r_min=0.0000000000000001
r_max=0.000000000001
num_of_steps = 6
r_step=(r_max-r_min)/num_of_steps

for A in Avals:
    r_and_c = cycle_central_charge(r_min, r_max, r_step, R, A, rapidityMax,rapidityMin,numPoints)
    print(r_and_c)
    save_charge(A,r_and_c)