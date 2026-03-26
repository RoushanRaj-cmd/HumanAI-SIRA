import torch
import torch.optim as optim
from core.model import compute_losses

def train_pinn(model, t_data, y_data, t_physics, epochs_adam=2000, epochs_lbfgs=1000, verbose=True):
    """
    Two-stage optimization process for Physics-Informed Neural Networks.
    Pre-trains with Adam, fine-tunes with L-BFGS.
    """
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    
    # Stage 1: Pre-training with Adam
    if verbose: print("Starting Adam Optimization...")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        loss_data, loss_physics, total_loss = compute_losses(model, t_data, y_data, t_physics)
        total_loss.backward()
        optimizer_adam.step()
        
        if verbose and epoch % 500 == 0:
            print(f"Adam Epoch {epoch}: Beta: {model.beta.item():.4f}, Gamma: {model.gamma.item():.4f}, Loss: {total_loss.item():.6f}")

    # Stage 2: Fine-tuning with L-BFGS
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1e-1, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100)
    
    if verbose: print("Starting L-BFGS Optimization...")
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_data, loss_physics, total_loss = compute_losses(model, t_data, y_data, t_physics)
        total_loss.backward()
        return total_loss

    for epoch in range(epochs_lbfgs):
        optimizer_lbfgs.step(closure)
        if verbose and epoch % 200 == 0:
            loss_data, loss_physics, total_loss = compute_losses(model, t_data, y_data, t_physics)
            print(f"L-BFGS Epoch {epoch}: Beta: {model.beta.item():.4f}, Gamma: {model.gamma.item():.4f}, Loss: {total_loss.item():.6f}")
            
    if verbose: print(f"Training Complete. Learned Beta: {model.beta.item():.4f}, Learned Gamma: {model.gamma.item():.4f}")
    return model
