import numpy as np
from core.solver import solve_sir

def generate_synthetic_data(S0, I0, R0, beta, gamma, t_max, num_points, noise_level=0.05, sparsity=1):
    """
    Generates realistic, noisy epidemic data from ideal SIR models.
    noise_level: proportion of standard deviation in Gaussian distribution (0.05 = 5%).
    sparsity: interval between saved data points (e.g. 7 = weekly reports).
    """
    t, y = solve_sir(S0, I0, R0, beta, gamma, t_max, num_points)
    
    S, I, R = y[:, 0], y[:, 1], y[:, 2]
    
    # Inject Gaussian Noise independently per curve
    S_noisy = S * (1 + np.random.normal(0, noise_level, len(S)))
    I_noisy = I * (1 + np.random.normal(0, noise_level, len(I)))
    R_noisy = R * (1 + np.random.normal(0, noise_level, len(R)))
    
    # Optional constraint bounds fixing values within [0.0, 1.0] if modeling normalized fractions
    S_noisy = np.clip(S_noisy, 0, 1)
    I_noisy = np.clip(I_noisy, 0, 1)
    R_noisy = np.clip(R_noisy, 0, 1)
    
    # Apply sparsity subsampling
    t_sparse = t[::sparsity]
    states_noisy = np.vstack([S_noisy[::sparsity], I_noisy[::sparsity], R_noisy[::sparsity]]).T
    
    return t_sparse, states_noisy
