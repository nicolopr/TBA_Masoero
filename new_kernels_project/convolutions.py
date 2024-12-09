import numpy as np
from definitions import nu, K

def initialize_epsilon(R, m, A, rapidityVals):
# Initialize epsilon values by ignoring convolution terms
    em2 = nu(rapidityVals,-2, R, m, A).astype(np.clongdouble)
    em1 = nu(rapidityVals,-1, R, m, A).astype(np.clongdouble)
    e0 = nu(rapidityVals,0, R, m, A).astype(np.clongdouble)
    e1= nu(rapidityVals,1, R, m, A).astype(np.clongdouble)
    e2= nu(rapidityVals,2, R, m, A).astype(np.clongdouble)
    return em2, em1, e0, e1, e2

# define the values of the rapidity allowed
def discretize_rapidity(rapidityMax,rapidityMin,numPoints):
    rapidityVals = np.linspace(rapidityMin, rapidityMax, numPoints)
    step = (rapidityMax - rapidityMin) / (numPoints - 1)
    return rapidityVals, step

# precompute the matrix theta-thetaprime in the convolution
def rapidity_diff(rapidityVals):
    return rapidityVals[:, np.newaxis] - rapidityVals[np.newaxis, :]

# Compute convolutions
def compute_convolution(k, Lm2, Lm1, L0, L1, L2, step, rapidityVals):
    Krapidities=rapidity_diff(rapidityVals)
    conv = K(Krapidities,-2,k).dot(Lm2)+K(Krapidities,-1,k).dot(Lm1)+K(Krapidities,0,k).dot(L0)+K(Krapidities,1,k).dot(L1)+K(Krapidities,2,k).dot(L2)
    return step * conv