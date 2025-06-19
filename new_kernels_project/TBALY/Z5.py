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
def phi(beta, k):
    sqrt3 = jnp.sqrt(3)
    denominator = 1 + 2 * jnp.cosh(2 * beta)

    def case_k0():
        return -2 * sqrt3 * (2 * jnp.cosh(beta)) / denominator

    def case_k1():
        return 2 * sqrt3 * jnp.exp(-9 / 5 * beta) / denominator

    def case_k2():
        return 2 * sqrt3 * jnp.exp(-3 / 5 * beta) / denominator

    def case_km1():
        return phi(-beta, 1)

    def case_km2():
        return phi(-beta, 2)

    return jax.lax.switch(k + 2, [case_km2, case_km1, case_k0, case_k1, case_k2])

@jax.jit
def compute_all_phis_on_rapidities(rapidityVals):
    ks = jnp.array([0, 1, 2, -1, -2])
    beta_diff = rapidityVals[:, None] - rapidityVals[None, :]  # shape (N, N)

    def compute_phi_for_k(k):
        flat = jax.vmap(phi, in_axes=(0, None))(beta_diff.ravel(), k)
        return flat.reshape(beta_diff.shape)  # shape (N, N)

    all_phis = jax.vmap(compute_phi_for_k)(ks)  # shape (5, N, N)
    return all_phis

# --- Vectorized Helpers ---
@jax.jit
def batch_cosh(r, rapidity):
    return r * jnp.cosh(rapidity)

# @jax.jit
# def compute_convolution(coeffs, lambdas, step, phi_0, phi_1, phi_2, phi_m1, phi_m2):
#     conv = (
#         coeffs[0] * phi_0.dot(lambdas[0])
#         + coeffs[1] * phi_1.dot(lambdas[1])
#         + coeffs[2] * phi_2.dot(lambdas[2])
#         + coeffs[3] * phi_m1.dot(lambdas[3])
#         + coeffs[4] * phi_m2.dot(lambdas[4])
#     )
#     return (step / (2 * jnp.pi)) * conv / 5
@jax.jit
def compute_lambda_from_L(L):  # L shape: (5, N)
    k = jnp.arange(5)
    n = jnp.arange(5)[:, None]
    # Create DFT matrix: shape (5, 5)
    F = jnp.exp(2j * jnp.pi * n * k / 5)
    # Matrix multiplication: shape (5, N)
    return F @ L

lambda_FFT=jax.jit(compute_lambda_from_L)

#alterative version with chatgpt improvement
@jax.jit
def compute_convolution(coeffs, lambdas, step, phis):
    # phis shape: (5, N, N), lambdas shape: (5, N)
    conv_terms = jnp.einsum("i,ijk,ik->j", coeffs, phis, lambdas)
    return (step / (2 * jnp.pi)) * conv_terms / 5


conv_jit = jax.jit(compute_convolution)


# --- TBA Loop ---
@jax.jit
def TBA_loop_jit(r, rapidityVals, step, phi_matrix):
    delta = 1.0
    iteration = 0
    epsilon1_old = batch_cosh(r, rapidityVals)
    epsilon_old = jnp.stack([epsilon1_old] * 5) #this is immutable, to be updated with epsilon_old.at[3].set(new_eps3)

    def cond_fun(loop_vars):
        delta, iteration, *_ = loop_vars
        return jnp.logical_and(delta > 1e-15, iteration < 5000)

    def body_fun(loop_vars): #to be fixed
        delta, iteration, eps_old, _ = loop_vars
        Ls = jnp.log1p(jnp.exp(-eps_old))
        conv = conv_jit(coeff, L, step, phi_matrix)
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
phi_matrix = compute_all_phis_on_rapidities(rapidityVals)
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
