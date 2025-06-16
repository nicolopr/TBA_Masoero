import numpy as np
import jax
import jax.numpy as jnp
import time

def discretize_rapidity(rapidityMax,rapidityMin,numPoints):
    rapidityVals = jnp.linspace(rapidityMin, rapidityMax, numPoints)
    step = (rapidityMax - rapidityMin) / (numPoints - 1)
    return rapidityVals, step

# Define Kernel
def phi(beta):
    sqrt3 = jnp.sqrt(3)
    denominator = 1 + 2 * jnp.cosh(2 * beta)
    return -2 * sqrt3 * (2 * jnp.cosh(beta)) / denominator

# def init_energy(r, rapidities):
#     return r * jnp.cosh(rapidities)

# Compute convolutions
def compute_convolution(coeff, Le, step, phi):
    conv = (
        coeff * phi.dot(Le)
    )
    return (step / (2 * jnp.pi)) * conv / 5

conv_jax=jax.jit(compute_convolution)

# Precompute phi matrices with vmap
def compute_phi_on_rapidities(rapidityVals):
    beta_diff = rapidityVals[:, jnp.newaxis] - rapidityVals[jnp.newaxis, :]  # Shape (numPoints, numPoints)
    batched_phi=jax.vmap(phi)
    phi1 = batched_phi(beta_diff)
    return phi1

# Iterative loop - to be jitted
def TBA_loop(r,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0
    deltaThreshold = 1e-15
    maxIterations = 5000

    #discretize rapidity space
    rapidityVals,step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi= compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    epsilon1Old= r * jnp.cosh(rapidityVals)
    # init_energy(r, rapidityVals)
    
    while delta > deltaThreshold and iteration < maxIterations:
        iteration += 1

        # Compute convolution terms
        lambda1 = jnp.log(1 + jnp.exp(-epsilon1Old))

        coeffs_list = [1, 1, 1, 1, 1]  # For conv1
        conv1 = conv_jax(coeffs_list[0], lambda1, step, phi)

        # Update epsilon values
        epsilon1New = r * jnp.cosh(rapidityVals) - conv1
        # Compute convergence delta
        delta= np.max(np.abs(epsilon1New - epsilon1Old))

        # Update old epsilon values
        epsilon1Old = epsilon1New

        # Print iteration info
        #print(f"Iteration {iteration}: delta = {delta}")
    return epsilon1New

def TBA_loop_jit(r,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0

    #discretize rapidity space
    rapidityVals, step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi= compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    def rcosh(rapidity):
        return r * jnp.cosh(rapidityVals)
    def lambdas(eps):
        return jnp.log(1 + jnp.exp(-eps))
    
    batch_cosh=jax.vmap(rcosh)
    batch_lambda=jax.vmap(lambdas)

    epsilon1Old= batch_cosh(rapidityVals)
    # init_energy(r, rapidityVals)
    initial_state = (delta, iteration, epsilon1Old, epsilon1Old)


    def cond_fun(loop_vars):
        deltaThreshold = 1e-15
        maxIterations = 5000
        delta=loop_vars[0]
        iteration=loop_vars[1]
        return jnp.logical_and(delta > deltaThreshold, iteration < maxIterations)

    def body_fun(loop_vars):
        delta, iteration, epsilon1Old, _ = loop_vars
        # Compute convolution terms
        lambda1 = batch_lambda(-epsilon1Old)

        coeffs_list = [1, 1, 1, 1, 1]  # For conv1
        conv1 = conv_jax(coeffs_list[0], lambda1, step, phi)

        # Update epsilon values
        epsilon1New = batch_cosh(rapidityVals) - conv1
        # Compute convergence delta
        new_delta= jnp.max(jnp.abs(epsilon1New - epsilon1Old))

        # Update old epsilon values
        epsilon1Old = epsilon1New
        return (new_delta, iteration+1, epsilon1New, epsilon1Old)


    final_state =jax.lax.while_loop(cond_fun, body_fun, initial_state)

    new_delta, iteration, epsilon1New, epsilon1Old = final_state
    return epsilon1New

def compute_central_charge(r, rapidityVals, epsilon, step):
    prefactor = (3 / jnp.pi**2) * r 

    # Define components for each epsilon
    cosh_vals = jnp.cosh(rapidityVals)
    log_terms = [
        (cosh_vals + jnp.cosh(rapidityVals )) * np.log(1 + np.exp(-epsilon))
    ]

    # Sum all terms for the central charge calculation
    total_sum = sum([jnp.sum(term) * step for term in log_terms])
    central_charge = prefactor * total_sum.real

    return central_charge

def cycle_central_charge(r_min, r_max, r_step):
    central_charges=[]
    r=r_min
    while r<=r_max:
        sumre=lambda n: jnp.real(n)+np.imag(n)
        e1=TBA_loop(r,rapidityMax,rapidityMin,numPoints)
        cc=compute_central_charge(r, rapidityvals, e1, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
        central_charges.append([r,cc])
        r=r+r_step
    return central_charges


rapidityMax = 20
rapidityMin = -20

numPoints = 1001

rapidityvals=discretize_rapidity(rapidityMax,rapidityMin, numPoints)[0]

r_min=0.00001
r_max=0.0001
num_of_steps = 2
r_step=(r_max-r_min)/num_of_steps

start = time.time()
r_and_c = cycle_central_charge(r_min, r_max, r_step)
end = time.time()
print("Python loop time: {:.4f} seconds".format(end - start))
print(r_and_c)