import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export

# Define kernels

def mod5(n):
    while n<-2:
        n+=5
    while n>2:
        n+=-5
    return n

def K(beta, i,j):
    def kern(beta):
        return -2*np.sqrt(3)/(5*np.pi)*np.cosh(beta/5)/(1+2*np.cosh(2*beta/5))
    if i==0:
        return kern(beta- 2j*j*np.pi)
    else:
        return K(beta,0,mod5(j-i))

# Define driving
def nu(beta,k, R, m, A):
    def nu_basic(beta):
        return R*(m*np.cosh(beta)+m**(1/5)*A*np.cosh(beta/5))
    return nu_basic(beta-2j*2*np.pi*k)






# def cycle_central_charge(r_min, r_max, r_step):
#     central_charges=[]
#     r=r_min
#     while r<=r_max:
#         print('valore di r:',r)
#         a = A*(r**(1/5))
#         e1,e2,e3,e4,e5=TBA_loop(r,a,rapidityMax,rapidityMin,numPoints)
#         cc=compute_central_charge(r, a, rapidityvals, e1, e2, e3, e4, e5, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
#         central_charges.append([r,cc])
#         r=r+r_step
#     return central_charges

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


r_min=0.0001
r_max=0.02
num_of_steps = 30
r_step=(r_max-r_min)/num_of_steps

#r_and_c = cycle_central_charge(r_min, r_max, r_step)
#print(r_and_c)
#save_charge(A,r_and_c)