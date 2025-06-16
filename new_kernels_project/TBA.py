from definitions import nu
from convolutions import initialize_epsilon, discretize_rapidity, compute_convolution
import numpy as np

# Iterative loop
def TBA_loop(R, m, A, rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0
    deltaThreshold = 1e-15
    maxIterations = 500

    #discretize rapidity space
    rapidityVals,step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)
    #initialise epsilon with the driving only
    epsilonm2Old, epsilonm1Old, epsilon0Old, epsilon1Old, epsilon2Old = initialize_epsilon(R, m, A, rapidityVals)

    while delta > deltaThreshold and iteration < maxIterations:
        iteration += 1

        # Compute lambdas
        Lm2 = np.log(1 + np.exp(-epsilonm2Old).astype(np.clongdouble)).astype(np.clongdouble)
        Lm1 = np.log(1 + np.exp(-epsilonm1Old).astype(np.clongdouble)).astype(np.clongdouble)
        L0 = np.log(1 + np.exp(-epsilon0Old).astype(np.clongdouble)).astype(np.clongdouble)
        L1 = np.log(1 + np.exp(-epsilon1Old).astype(np.clongdouble)).astype(np.clongdouble)
        L2 = np.log(1 + np.exp(-epsilon2Old).astype(np.clongdouble)).astype(np.clongdouble)

        # Compute convolutions
        
        convm2 = compute_convolution(-2, Lm2, Lm1, L0, L1, L2, step, rapidityVals)
        convm1 = compute_convolution(-1, Lm2, Lm1, L0, L1, L2, step, rapidityVals)
        conv0 = compute_convolution(0, Lm2, Lm1, L0, L1, L2, step, rapidityVals)
        conv1 = compute_convolution(1, Lm2, Lm1, L0, L1, L2, step, rapidityVals)
        conv2 = compute_convolution(2, Lm2, Lm1, L0, L1, L2, step, rapidityVals)

        # Update epsilon values - not sure about sign in convolution terms, in old code there was a - but I am not sure why
        epsilonm2New = nu(rapidityVals,-2,R, m, A) + convm2
        epsilonm1New = nu(rapidityVals,-1,R, m, A) + convm1
        epsilon0New = nu(rapidityVals,0,R, m, A) + conv0
        epsilon1New = nu(rapidityVals,1,R, m, A) + conv1
        epsilon2New = nu(rapidityVals,2,R, m, A) + conv2

        # Compute convergence delta
        deltam2 = np.max(np.abs(epsilonm2New - epsilonm2Old))
        deltam1 = np.max(np.abs(epsilonm1New - epsilonm1Old))
        delta0 = np.max(np.abs(epsilon0New - epsilon0Old))
        delta1 = np.max(np.abs(epsilon1New - epsilon1Old))
        delta2 = np.max(np.abs(epsilon2New - epsilon2Old))
        delta = max(deltam1, deltam2, delta2, delta1, delta0)

        # Update old epsilon values
        epsilonm1Old = epsilonm1New
        epsilonm2Old = epsilonm2New
        epsilon0Old = epsilon0New
        epsilon1Old = epsilon1New
        epsilon2Old = epsilon2New

        # Print iteration info
    print(f"Iteration {iteration}: delta = {delta}")
    return epsilonm2New, epsilonm1New, epsilon0New, epsilon1New, epsilon2New