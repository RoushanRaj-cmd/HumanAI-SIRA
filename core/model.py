import torch
import torch.nn as nn

class SIR_PINN(nn.Module):
    def __init__(self, hidden_layers=3, nodes=32):
        super(SIR_PINN, self).__init__()
        layers = [nn.Linear(1, nodes), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(nodes, nodes), nn.Tanh()])
        layers.append(nn.Linear(nodes, 3))
        self.net = nn.Sequential(*layers)
        
        self._beta = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self._gamma = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        
        self.log_var_data = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.log_var_physics = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    @property
    def beta(self): return torch.exp(self._beta)

    @property
    def gamma(self): return torch.exp(self._gamma)

    def forward(self, t):
        # Normalize t by 100.0 to prevent Tanh vanishing gradients (dead neurons) on large time steps
        return self.net(t / 100.0)

class SEIR_PINN(nn.Module):
    def __init__(self, hidden_layers=3, nodes=32):
        super(SEIR_PINN, self).__init__()
        layers = [nn.Linear(1, nodes), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(nodes, nodes), nn.Tanh()])
        layers.append(nn.Linear(nodes, 4)) # S, E, I, R
        self.net = nn.Sequential(*layers)
        
        self._beta = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self._sigma = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self._gamma = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        
        self.log_var_data = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.log_var_physics = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    @property
    def beta(self): return torch.exp(self._beta)
    @property
    def sigma(self): return torch.exp(self._sigma)
    @property
    def gamma(self): return torch.exp(self._gamma)

    def forward(self, t):
        # Normalize t by 100.0 to prevent Tanh vanishing gradients
        return self.net(t / 100.0)

def compute_losses(model, t_data, y_data, t_physics):
    y_pred = model(t_data)
    loss_data = torch.mean((y_pred - y_data)**2)
    
    # Enforce hard penalty for Initial Conditions to anchor t=0 properly natively
    loss_ic = torch.mean((y_pred[0] - y_data[0])**2)
    
    t_physics.requires_grad = True
    u = model(t_physics)
    
    if isinstance(model, SIR_PINN):
        S, I, R = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        dSdt = torch.autograd.grad(S, t_physics, torch.ones_like(S), create_graph=True)[0]
        dIdt = torch.autograd.grad(I, t_physics, torch.ones_like(I), create_graph=True)[0]
        dRdt = torch.autograd.grad(R, t_physics, torch.ones_like(R), create_graph=True)[0]
        
        f_S = dSdt + model.beta * S * I
        f_I = dIdt - model.beta * S * I + model.gamma * I
        f_R = dRdt - model.gamma * I
        
        loss_physics = torch.mean(f_S**2 + f_I**2 + f_R**2)
        
    elif isinstance(model, SEIR_PINN):
        S, E, I, R = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]
        dSdt = torch.autograd.grad(S, t_physics, torch.ones_like(S), create_graph=True)[0]
        dEdt = torch.autograd.grad(E, t_physics, torch.ones_like(E), create_graph=True)[0]
        dIdt = torch.autograd.grad(I, t_physics, torch.ones_like(I), create_graph=True)[0]
        dRdt = torch.autograd.grad(R, t_physics, torch.ones_like(R), create_graph=True)[0]
        
        f_S = dSdt + model.beta * S * I
        f_E = dEdt - model.beta * S * I + model.sigma * E
        f_I = dIdt - model.sigma * E + model.gamma * I
        f_R = dRdt - model.gamma * I
        
        loss_physics = torch.mean(f_S**2 + f_E**2 + f_I**2 + f_R**2)
    
    # Adaptive Log-Variance dynamically weighting the inverse problem (Phase 2 spec)
    total_loss = (torch.exp(-model.log_var_data) * loss_data + model.log_var_data) + \
                 (torch.exp(-model.log_var_physics) * loss_physics + model.log_var_physics) + \
                 (100.0 * loss_ic)
    
    return loss_data, loss_physics, total_loss
