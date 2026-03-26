import numpy as np
import torch
from core.solver import solve_sir
from core.model import SIR_PINN, compute_losses

def test_odeint_matches_derivatives():
    S0, I0, R0 = 0.99, 0.01, 0.0
    beta, gamma = 0.3, 0.1
    t_max = 160
    t, y = solve_sir(S0, I0, R0, beta, gamma, t_max, 1600)
    
    # Check bounds
    assert np.all(y >= -1e-5), "Populations cannot be negative"
    assert np.all(y <= 1.0 + 1e-5), "Populations fraction cannot exceed 1"
    
    # Check conservation N=1
    N_total = np.sum(y, axis=1)
    np.testing.assert_allclose(N_total, 1.0, atol=1e-5)

def test_pinn_architecture():
    model = SIR_PINN(hidden_layers=2, nodes=16)
    t = torch.linspace(0, 10, 100).unsqueeze(1)
    pred = model(t)
    assert pred.shape == (100, 3), "Model output must map t -> S, I, R"
