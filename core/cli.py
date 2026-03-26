import argparse
import torch
from core.model import SIR_PINN
from core.trainer import train_pinn
from data.generator import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="SIRA1 PINN Epidemic Training CLI")
    parser.add_argument("--noise", type=float, default=0.05, help="Gaussian noise level (e.g., 0.05 for 5%)")
    parser.add_argument("--epochs-adam", type=int, default=2000, help="Adam Pre-training epochs")
    parser.add_argument("--epochs-lbfgs", type=int, default=500, help="L-BFGS Fine-tuning epochs")
    args = parser.parse_args()

    print(f"Generating synthetic observations with {args.noise*100}% noise...")
    # True parameters
    true_beta = 0.3
    true_gamma = 0.1
    t_sparse, y_noisy = generate_synthetic_data(0.99, 0.01, 0.0, true_beta, true_gamma, t_max=160, num_points=160, noise_level=args.noise, sparsity=1)
    
    # Convert vectors mapped to FloatTensors
    t_data = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(1)
    y_data = torch.tensor(y_noisy, dtype=torch.float32)
    t_physics = torch.linspace(0, 160, 400).unsqueeze(1)
    
    print("Initializing Physics-Informed Neural Network...")
    model = SIR_PINN(hidden_layers=3, nodes=32)
    
    print(f"True Parameters -> Beta: {true_beta}, Gamma: {true_gamma}")
    model = train_pinn(model, t_data, y_data, t_physics, epochs_adam=args.epochs_adam, epochs_lbfgs=args.epochs_lbfgs)
    
    # Error computations
    beta_err = abs(model.beta.item() - true_beta) / true_beta
    gamma_err = abs(model.gamma.item() - true_gamma) / true_gamma
    print(f"Estimation Errors -> Beta: {beta_err*100:.2f}%, Gamma: {gamma_err*100:.2f}%")

if __name__ == "__main__":
    main()
