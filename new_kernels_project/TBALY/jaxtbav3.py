import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Enable float64 precision for numerical stability
jax.config.update("jax_enable_x64", True)

# --- Discretization ---
def discretize_rapidity(rapidityMax, rapidityMin, numPoints):
    rapidityVals = jnp.linspace(rapidityMin, rapidityMax, numPoints)
    step = (rapidityMax - rapidityMin) / (numPoints - 1)
    return rapidityVals, step

# --- Kernel ---
def phi(beta):
    sqrt3 = jnp.sqrt(3)
    denominator = 1 + 2 * jnp.cosh(2 * beta)
    return -2 * sqrt3 * (2 * jnp.cosh(beta)) / denominator

@jax.jit
def compute_phi_on_rapidities(rapidityVals):
    beta_diff = rapidityVals[:, None] - rapidityVals[None, :]
    return jax.vmap(phi)(beta_diff)

# --- Vectorized Helpers ---
@jax.jit
def batch_cosh(r, rapidity):
    return r * jnp.cosh(rapidity)

@jax.jit
def compute_convolution(Le, step, phi_matrix):
    return (step / (2 * jnp.pi)) * phi_matrix.dot(Le)

conv_jit = jax.jit(compute_convolution)

# --- TBA Loop ---
@jax.jit
def TBA_loop_jit(r, rapidityVals, step, phi_matrix):
    delta = 1.0
    iteration = 0
    epsilon_old = batch_cosh(r, rapidityVals)

    def cond_fun(loop_vars):
        delta, iteration, *_ = loop_vars
        return jnp.logical_and(delta > 1e-15, iteration < 5000)

    def body_fun(loop_vars):
        delta, iteration, eps_old, _ = loop_vars
        L = jnp.log1p(jnp.exp(-eps_old))
        conv = conv_jit(L, step, phi_matrix)
        eps_new = batch_cosh(r, rapidityVals) - conv
        new_delta = jnp.max(jnp.abs(eps_new - eps_old))
        return new_delta, iteration + 1, eps_new, eps_old

    final_state = jax.lax.while_loop(cond_fun, body_fun, (delta, iteration, epsilon_old, epsilon_old))
    return final_state[2]  # epsilon_new

# --- Central Charge ---
@jax.jit
def compute_central_charge(r, cosh_vals, epsilon, step):
    prefactor = (3 / jnp.pi**2) * r
    integrand = cosh_vals * jax.nn.softplus(-epsilon)
    return prefactor * jnp.sum(integrand) * step

# --- Full Simulation ---
def cycle_central_charge(r_min, r_max, r_step, rapidityVals, step, phi_matrix, cosh_vals):
    r_vals = jnp.arange(r_min, r_max + r_step, r_step)
    results = []

    for r in r_vals:
        eps = TBA_loop_jit(r, rapidityVals, step, phi_matrix)
        cc = compute_central_charge(r, cosh_vals, eps, step)
        results.append([r, cc])

    return results

# --- Parameters ---
rapidityMax = 20
rapidityMin = -20
numPoints = 10001
r_min = 1e-11
r_max = 1e-7
num_steps = 2
r_step = (r_max - r_min) / num_steps

# --- Precompute values ---
rapidityVals, step = discretize_rapidity(rapidityMax, rapidityMin, numPoints)
phi_matrix = compute_phi_on_rapidities(rapidityVals)
cosh_vals = jnp.cosh(rapidityVals)

# --- Run Simulation ---
start = time.time()
r_and_c = cycle_central_charge(r_min, r_max, r_step, rapidityVals, step, phi_matrix, cosh_vals)
end = time.time()

# --- Output ---
print("Simulation Time: {:.4f} seconds".format(end - start))
for r, cc in r_and_c:
    print(f"r = {r:.2e}, Central Charge = {cc:.10f}")

# --- Plotting ---
r_vals, cc_vals = zip(*r_and_c)
plt.plot(r_vals, cc_vals, marker='o')
plt.xscale('log')
plt.xlabel("r")
plt.ylabel("Central Charge")
plt.title("Central Charge vs r")
plt.grid(True)
plt.show()
