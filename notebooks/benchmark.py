import torch
from core.model import SIR_PINN
from core.trainer import train_pinn
from data.generator import generate_synthetic_data

def run_benchmarks():
    noise_levels = [0.01, 0.05, 0.10, 0.15]
    true_beta, true_gamma = 0.3, 0.1
    
    print("Running Robustness Benchmark...")
    for noise in noise_levels:
        print(f"\n--- Testing Noise Level: {noise*100}% ---")
        t_sparse, y_noisy = generate_synthetic_data(0.99, 0.01, 0.0, true_beta, true_gamma, 160, 160, noise, sparsity=1)
        t_data = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(1)
        y_data = torch.tensor(y_noisy, dtype=torch.float32)
        t_physics = torch.linspace(0, 160, 400).unsqueeze(1)
        
        model = SIR_PINN(hidden_layers=3, nodes=32)
        # Using less epochs to speed up benchmark run locally
        train_pinn(model, t_data, y_data, t_physics, epochs_adam=1000, epochs_lbfgs=50, verbose=False)
        
        b_err = abs(model.beta.item() - true_beta)/true_beta * 100
        g_err = abs(model.gamma.item() - true_gamma)/true_gamma * 100
        print(f"Noise {noise*100}% Error -> Beta: {b_err:.2f}%, Gamma: {g_err:.2f}%")

if __name__ == "__main__":
    run_benchmarks()
