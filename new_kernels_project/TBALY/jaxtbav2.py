import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

def discretize_rapidity(rapidityMax,rapidityMin,numPoints):
    rapidityVals = jnp.linspace(rapidityMin, rapidityMax, numPoints)
    step = (rapidityMax - rapidityMin) / (numPoints - 1)
    return rapidityVals, step

# Define Kernel
def phi(beta):
    sqrt3 = jnp.sqrt(3)
    denominator = 1 + 2 * jnp.cosh(2 * beta)
    return -2 * sqrt3 * (2 * jnp.cosh(beta)) / denominator

# Compute convolutions
def compute_convolution(Le, step, phi):
    conv = (
        phi.dot(Le)
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

def TBA_loop(r,rapidityMax,rapidityMin,numPoints):
    # Initialize convergence parameters
    delta = 1
    iteration = 0
    deltaThreshold = 1e-15
    maxIterations = 100

    #discretize rapidity space
    rapidityVals,step=discretize_rapidity(rapidityMax,rapidityMin,numPoints)

    #compute kernels
    phi_0 = compute_phi_on_rapidities(rapidityVals)
    #initialize epsilon to their free value
    epsilon1Old= batch_cosh(r, rapidityVals)

    while iteration < maxIterations:
        iteration += 1

        # Compute convolution terms
        L1 = jnp.log(1 + jnp.exp(-epsilon1Old))

        # Compute lambda functions
        lambda1 = L1 

        conv1 = compute_convolution(lambda1, step, phi_0)

        # Update epsilon values
        epsilon1New = batch_cosh(r,rapidityVals) - conv1

        # Compute convergence delta
        delta = jnp.max(jnp.abs(epsilon1New - epsilon1Old))

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
        lambda1 = batch_lambda(epsilon1Old)

        conv1 = conv_jax(lambda1, step, phi)

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
    cosh_vals = batch_cosh(1,rapidityVals)
    integrand = cosh_vals * jax.nn.softplus(-epsilon)
    return prefactor * jnp.sum(integrand) * step

def cond_fun(loop_vars):
        deltaThreshold = 1e-15
        maxIterations = 5000
        delta=loop_vars[0]
        iteration=loop_vars[1]
        return jnp.logical_and(delta > deltaThreshold, iteration < maxIterations)


def cycle_central_charge(r_min, r_max, r_step):
    central_charges=[]
    r=r_min
    while r<=r_max:
        e1=TBA_loop_jit(r,rapidityMax,rapidityMin,numPoints)
        # plot_Y0(rapvals[0],logY0)
        # plot_epsilon_real(rapvals[0],e1,e2,e3,e4,e5)
        cc=compute_central_charge(r, rapidityvals, e1, discretize_rapidity(rapidityMax, rapidityMin, numPoints)[1])
        central_charges.append([r,cc])
        r=r+r_step
    return central_charges

rapidityMax = 20
rapidityMin = -20

numPoints = 10001

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
