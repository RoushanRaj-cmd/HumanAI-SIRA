import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.generator import generate_synthetic_data
from core.model import SIR_PINN
from core.trainer import train_pinn
import torch

st.set_page_config(page_title="SIRA1 Epidemic PINN", layout="wide")

st.title("🦠 SIRA1: PINN Epidemic Inference")
st.markdown("Physics-Informed Neural Network tracking Epidemic Data through automatic differentiation.")

st.sidebar.header("Simulator Configurations")
true_beta = st.sidebar.slider("True Beta (Infection Rate)", 0.05, 1.0, 0.3)
true_gamma = st.sidebar.slider("True Gamma (Recovery Rate)", 0.05, 1.0, 0.1)
noise = st.sidebar.slider("Gaussian Noise Level", 0.0, 0.2, 0.05)

st.sidebar.header("PINN Training Parameters")
st.sidebar.markdown("PINNs require high epoch counts to map physics losses effectively. Increase these if errors are high!")
epochs_adam = st.sidebar.slider("Adam Epochs", 500, 5000, 2000, step=100)
epochs_lbfgs = st.sidebar.slider("L-BFGS Epochs", 10, 500, 50, step=10)

if st.sidebar.button("Run PINN Inference"):
    with st.spinner(f"Training PINN with {epochs_adam} Adam iterations and {epochs_lbfgs} L-BFGS iterations..."):
        t_sparse, y_noisy = generate_synthetic_data(0.99, 0.01, 0.0, true_beta, true_gamma, 100, 100, noise, 1)
        
        t_data = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(1)
        y_data = torch.tensor(y_noisy, dtype=torch.float32)
        # Physics grid densely packed
        t_physics = torch.linspace(0, 100, 200).unsqueeze(1)
        
        model = SIR_PINN(hidden_layers=3, nodes=32)
        # Train locally using configured epochs
        train_pinn(model, t_data, y_data, t_physics, epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs, verbose=False)
        
        learned_beta = model.beta.item()
        learned_gamma = model.gamma.item()
        
        col1, col2 = st.columns(2)
        col1.metric("Learned Beta", f"{learned_beta:.4f}", f"{(learned_beta - true_beta)/true_beta * 100:.2f}% Error", delta_color="inverse")
        col2.metric("Learned Gamma", f"{learned_gamma:.4f}", f"{(learned_gamma - true_gamma)/true_gamma * 100:.2f}% Error", delta_color="inverse")
        
        # Inference Plotting
        t_fine = torch.linspace(0, 100, 200).unsqueeze(1)
        y_pred = model(t_fine).detach().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(t_sparse, y_noisy[:, 0], color='blue', alpha=0.3, label='S Noisy')
        ax.scatter(t_sparse, y_noisy[:, 1], color='red', alpha=0.3, label='I Noisy')
        ax.scatter(t_sparse, y_noisy[:, 2], color='green', alpha=0.3, label='R Noisy')
        
        ax.plot(t_fine.numpy(), y_pred[:, 0], 'b-', linewidth=2, label='S Predicted')
        ax.plot(t_fine.numpy(), y_pred[:, 1], 'r-', linewidth=2, label='I Predicted')
        ax.plot(t_fine.numpy(), y_pred[:, 2], 'g-', linewidth=2, label='R Predicted')
        
        ax.set_title("Actual vs. Predicted Compartmental Curves (MSE + PINN Loss)")
        ax.legend()
        st.pyplot(fig)
        
        st.success("Training Complete!")
