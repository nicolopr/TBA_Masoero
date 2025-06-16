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
    return (step / (2 * jnp.pi)) * conv

conv_jax=jax.jit(compute_convolution)

def rcosh(r,rapidity):
        return r * jnp.cosh(rapidity)
def lambdas(eps):
        return jax.nn.softplus(-eps)
    
batch_cosh=jax.vmap(rcosh, in_axes=(None,0))
batch_lambda=jax.vmap(lambdas)

# Precompute phi matrices with vmap
def compute_phi_on_rapidities(rapidityVals):
    beta_diff = rapidityVals[:, jnp.newaxis] - rapidityVals[jnp.newaxis, :]  # Shape (numPoints, numPoints)
    batched_phi=jax.vmap(phi)
    phi1 = batched_phi(beta_diff)
    return phi1

# Iterative loop

def TBA_loop_jit(r,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0

    #discretize rapidity space
    rapidityVals, step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi= compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    epsilon1Old= batch_cosh(r, rapidityVals)
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
        epsilon1New = batch_cosh(r,rapidityVals) - conv1
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
    cosh_vals = batch_cosh(r,rapidityVals)
    log_terms = [
        (cosh_vals + batch_cosh(r, rapidityVals )) * batch_lambda(-epsilon)
    ]
    # Sum all terms for the central charge calculation
    def compute_scaled_sum(term):
        return jnp.sum(term) * step

    # Convert list to array if possible, otherwise use lax.map or vmap on list
    central_charge = jnp.sum(jax.vmap(compute_scaled_sum)(jnp.stack(log_terms)))
    return central_charge

def cond_fun(loop_vars):
        deltaThreshold = 1e-15
        maxIterations = 5000
        delta=loop_vars[0]
        iteration=loop_vars[1]
        return jnp.logical_and(delta > deltaThreshold, iteration < maxIterations)

def cycle_central_charge(r_min, r_max, num_steps):
    def scan_step(carry, r):
        rapidityvals, step = carry  # carry = (rapidityvals, step)
    
        e1 = TBA_loop_jit(r, rapidityMax, rapidityMin, numPoints)
        cc = compute_central_charge(r, rapidityvals, e1, step)
        print(cc)
    
        return carry, jnp.array([r, cc])  # carry unchanged, output is [r, cc]
    rs = jnp.linspace(r_min, r_max, num_steps)
    rapidityvals, step = discretize_rapidity(rapidityMax, rapidityMin, numPoints)

    # Run the scan
    _, central_charges = jax.lax.scan(scan_step, (rapidityvals, step), rs)

    return central_charges


rapidityMax = 20
rapidityMin = -20

numPoints = 1001

rapidityvals=discretize_rapidity(rapidityMax,rapidityMin, numPoints)[0]

r_min=0.00001
r_max=0.0001
num_of_steps = 2

start = time.time()
r_and_c = cycle_central_charge(r_min, r_max, num_of_steps)
end = time.time()
print("Python loop time: {:.4f} seconds".format(end - start))
print(r_and_c)