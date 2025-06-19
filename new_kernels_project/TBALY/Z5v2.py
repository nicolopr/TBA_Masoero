import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial

# Enable float64 precision for numerical stability
jax.config.update("jax_enable_x64", True)

# --- Discretization ---
def discretize_rapidity(rapidity_max, rapidity_min, num_points):
    """Create discretized rapidity grid and step size."""
    rapidity_vals = jnp.linspace(rapidity_min, rapidity_max, num_points)
    step = (rapidity_max - rapidity_min) / (num_points - 1)
    return rapidity_vals, step

# --- Kernel Functions ---
@jax.jit
def phi_kernel(beta, k):
    """
    Compute the TBA kernel phi_k(beta).
    This appears to be for a model with 5 particle types.
    """
    sqrt3 = jnp.sqrt(3.0)
    
    # Handle numerical stability
    beta = jnp.where(jnp.abs(beta) < 1e-15, 1e-15, beta)
    denominator = 1.0 + 2.0 * jnp.cosh(2.0 * beta)
    
    def case_k0():
        return -2.0 * sqrt3 * (2.0 * jnp.cosh(beta)) / denominator
    
    def case_k1():
        return 2.0 * sqrt3 * jnp.exp(-1.8 * beta) / denominator  # -9/5 = -1.8
    
    def case_k2():
        return 2.0 * sqrt3 * jnp.exp(-0.6 * beta) / denominator  # -3/5 = -0.6
    
    def case_km1():
        return 2.0 * sqrt3 * jnp.exp(1.8 * beta) / denominator   # phi(-beta, 1)
    
    def case_km2():
        return 2.0 * sqrt3 * jnp.exp(0.6 * beta) / denominator   # phi(-beta, 2)
    
    # Map k to [0,4] range for switch
    return jax.lax.switch(k + 2, [case_km2, case_km1, case_k0, case_k1, case_k2])

@jax.jit
def compute_kernel_matrix(rapidity_vals):
    """Compute the full kernel matrix for all rapidity differences and all k values."""
    ks = jnp.array([-2, -1, 0, 1, 2])
    beta_diff = rapidity_vals[:, None] - rapidity_vals[None, :]
    
    def compute_phi_for_k(k):
        return jax.vmap(jax.vmap(phi_kernel, in_axes=(0, None)), in_axes=(1, None))(beta_diff, k)
    
    return jax.vmap(compute_phi_for_k)(ks)  # shape (5, N, N)

# --- FFT-based convolution (following your original approach) ---
@jax.jit
def compute_lambda_from_L(L):
    """Compute lambda using FFT as in original code."""
    k = jnp.arange(5)
    n = jnp.arange(5)[:, None]
    F = jnp.exp(2j * jnp.pi * n * k / 5)
    return F @ L  # shape (5, N)

@jax.jit
def compute_all_convolutions(lambdas, step, phis):
    """Compute convolutions following the original approach."""
    def exp_factor(n):
        return jnp.exp(2j * jnp.pi * n / 5)

    ks = jnp.array([0, 1, 2, -1, -2])
    coeffs_list = jnp.stack([
        jnp.ones(5, dtype=jnp.complex128),
        jnp.array([exp_factor(k) for k in ks]),
        jnp.array([exp_factor(2 * k) for k in ks]),
        jnp.array([exp_factor(-k) for k in ks]),
        jnp.array([exp_factor(-2 * k) for k in ks]),
    ])  # shape (5, 5)

    def single_convolution(coeffs):
        return (step / (2 * jnp.pi)) * jnp.tensordot(coeffs, jnp.einsum('ijk,ik->j', phis, lambdas), axes=1) / 5

    return jax.vmap(single_convolution)(coeffs_list)  # shape (5, N)

# --- TBA Solver ---
@jax.jit
def tba_iteration_step(epsilon_old, r, rapidity_vals, phi_matrix, step):
    """Single TBA iteration step following the original logic."""
    # Compute L values with clipping for stability
    epsilon_clipped = jnp.clip(epsilon_old, -50, 50)
    L = jnp.log1p(jnp.exp(-epsilon_clipped))
    
    # Compute lambdas using FFT
    lambdas = compute_lambda_from_L(L)
    
    # Compute convolutions
    conv = compute_all_convolutions(lambdas, step, phi_matrix)
    
    # Compute new epsilon (driving term - convolution)
    cosh_vals = jnp.cosh(rapidity_vals)
    driving_term = r * cosh_vals
    epsilon_new = driving_term - conv.real  # Take real part as in original
    
    return epsilon_new

@jax.jit
def solve_tba_equations(r, rapidity_vals, phi_matrix, step, 
                       max_iterations=5000, tolerance=1e-12):
    """Solve TBA equations iteratively."""
    N = len(rapidity_vals)
    
    # Initialize with driving term for all particle types
    cosh_vals = jnp.cosh(rapidity_vals)
    epsilon_init = jnp.stack([r * cosh_vals] * 5)  # Shape (5, N)
    
    def cond_fun(carry):
        iteration, epsilon_old, epsilon_new, delta = carry
        return jnp.logical_and(delta > tolerance, iteration < max_iterations)
    
    def body_fun(carry):
        iteration, epsilon_old, epsilon_new, _ = carry
        epsilon_next = tba_iteration_step(epsilon_new, r, rapidity_vals, phi_matrix, step)
        delta = jnp.max(jnp.abs(epsilon_next - epsilon_new))
        return iteration + 1, epsilon_new, epsilon_next, delta
    
    # First iteration
    epsilon_new = tba_iteration_step(epsilon_init, r, rapidity_vals, phi_matrix, step)
    delta = jnp.max(jnp.abs(epsilon_new - epsilon_init))
    initial_state = (0, epsilon_init, epsilon_new, delta)
    
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    
    return final_state[2]  # Return final epsilon values

# --- Central Charge Calculation ---
@jax.jit
def compute_central_charge(r, rapidity_vals, epsilon_values, step):
    """
    Compute central charge. For a model with expected c=2, the formula should be:
    c = (3/π²) * r * ∫ cosh(θ) * sum_i L_i(θ) dθ
    
    This is based on the general TBA formula for central charge.
    """
    cosh_vals = jnp.cosh(rapidity_vals)
    epsilon_clipped = jnp.clip(epsilon_values, -50, 50)
    L_values = jnp.log1p(jnp.exp(-epsilon_clipped))
    
    # Sum L values over all particle types
    total_L = jnp.sum(L_values, axis=0)  # Sum over first axis (particle types)
    
    # Integrate cosh(θ) * total_L(θ)
    integrand = cosh_vals * total_L
    integral = jnp.sum(integrand) * step
    
    # Prefactor - this may need adjustment based on the specific model
    prefactor = 3.0 / (jnp.pi**2) * r
    
    return prefactor * integral

# --- Main Simulation ---
def main():
    # Parameters - starting conservative
    rapidity_max = 12.0
    rapidity_min = -12.0
    num_points = 2001
    
    # Temperature range - focusing on the regime where c should approach 2
    r_min = 1e-8
    r_max = 1e-4
    num_r_points = 8
    
    print("Setting up TBA calculation...")
    print(f"Rapidity range: [{rapidity_min}, {rapidity_max}] with {num_points} points")
    print(f"Temperature range: r ∈ [{r_min:.2e}, {r_max:.2e}] with {num_r_points} points")
    print("Expected central charge: c = 2")
    
    # Setup discretization
    rapidity_vals, step = discretize_rapidity(rapidity_max, rapidity_min, num_points)
    print(f"Rapidity step size: {step:.6f}")
    
    # Precompute kernel matrix
    print("\nPrecomputing kernel matrix...")
    start_time = time.time()
    phi_matrix = compute_kernel_matrix(rapidity_vals)
    print(f"Kernel computation time: {time.time() - start_time:.4f} seconds")
    print(f"Kernel matrix shape: {phi_matrix.shape}")
    
    # Check kernel properties
    kernel_max = jnp.max(jnp.abs(phi_matrix))
    print(f"Max kernel magnitude: {kernel_max:.3e}")
    
    # Generate r values
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), num_r_points)
    
    # Run simulation
    print(f"\nRunning TBA simulation...")
    results_list = []
    
    for i, r in enumerate(r_values):
        print(f"Processing r = {r:.3e} ({i+1}/{len(r_values)})")
        
        try:
            start_time = time.time()
            epsilon = solve_tba_equations(r, rapidity_vals, phi_matrix, step)
            solve_time = time.time() - start_time
            
            cc = compute_central_charge(r, rapidity_vals, epsilon, step)
            
            results_list.append((float(r), float(cc)))
            print(f"  Central charge: {cc:.6f} (solved in {solve_time:.3f}s)")
            
            # Check for reasonable convergence
            if cc < 0 or cc > 10:
                print(f"  Warning: Unusual central charge value!")
                
        except Exception as e:
            print(f"  Error: {e}")
            results_list.append((float(r), float('nan')))
    
    # Process results
    r_vals = [r for r, cc in results_list]
    cc_vals = [cc for r, cc in results_list]
    
    print(f"\nResults Summary:")
    print("-" * 60)
    c_theory = 2.0
    
    for r, cc in results_list:
        if not np.isnan(cc):
            error = abs(cc - c_theory) / c_theory * 100
            print(f"r = {r:.3e}, c = {cc:.6f}, error = {error:.2f}%")
        else:
            print(f"r = {r:.3e}, c = NaN (failed)")
    
    # Check trend - central charge should approach 2 as r decreases
    valid_results = [(r, cc) for r, cc in results_list if not np.isnan(cc)]
    if len(valid_results) > 1:
        r_sorted, cc_sorted = zip(*sorted(valid_results))
        print(f"\nTrend analysis:")
        print(f"Smallest r: {r_sorted[0]:.3e} -> c = {cc_sorted[0]:.6f}")
        print(f"Largest r:  {r_sorted[-1]:.3e} -> c = {cc_sorted[-1]:.6f}")
        
        if len(valid_results) >= 3:
            # Check if approaching c=2
            final_c = cc_sorted[0]  # Smallest r should give best approximation
            print(f"Best estimate: c = {final_c:.6f}")
            print(f"Error from expected c=2: {abs(final_c - 2.0):.6f}")
    
    # Plotting
    valid_r = [r for r, cc in valid_results]
    valid_cc = [cc for r, cc in valid_results]
    
    if valid_r:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.semilogx(valid_r, valid_cc, 'bo-', markersize=8, linewidth=2, label='TBA calculation')
        plt.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Theory (c = 2)')
        plt.xlabel('Temperature parameter r')
        plt.ylabel('Central Charge')
        plt.title('Central Charge vs Temperature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        errors = [abs(cc - 2.0) for cc in valid_cc]
        plt.loglog(valid_r, errors, 'ro-', markersize=6, linewidth=2)
        plt.xlabel('Temperature parameter r')
        plt.ylabel('|c - 2|')
        plt.title('Error in Central Charge')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return r_vals, cc_vals

if __name__ == "__main__":
    main()