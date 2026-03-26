from scipy.integrate import odeint
import numpy as np

def solve_sir(S0, I0, R0, beta, gamma, t_max, num_points):
    """
    Solving strictly the standard SIR system numerically using scipy.odeint.
    Provides baselines measuring residuals of ODEs.
    """
    t = np.linspace(0, t_max, num_points)
    
    def sir_deriv(y, t, beta, gamma):
        S, I, R = y
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]
        
    y = odeint(sir_deriv, [S0, I0, R0], t, args=(beta, gamma))
    return t, y

def solve_seir(S0, E0, I0, R0, beta, sigma, gamma, t_max, num_points):
    """
    Extended architecture mapping SEIR numerically.
    """
    t = np.linspace(0, t_max, num_points)
    
    def seir_deriv(y, t, beta, sigma, gamma):
        S, E, I, R = y
        return [
            -beta * S * I,
            beta * S * I - sigma * E,
            sigma * E - gamma * I,
            gamma * I
        ]
        
    y = odeint(seir_deriv, [S0, E0, I0, R0], t, args=(beta, sigma, gamma))
    return t, y
