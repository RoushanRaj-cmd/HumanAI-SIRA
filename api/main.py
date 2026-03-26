from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
import pandas as pd
import numpy as np
import io
import math
from core.model import SIR_PINN
from core.trainer import train_pinn

app = FastAPI(title="SIRA1 PINN Epidemic Inference API")

class TimeSeriesData(BaseModel):
    t: List[float]
    S: List[float]
    I: List[float]
    R: List[float]
    epochs_adam: int = 500
    epochs_lbfgs: int = 20

@app.post("/infer_parameters")
def infer_parameters(data: TimeSeriesData):
    try:
        t_data = torch.tensor(data.t, dtype=torch.float32).unsqueeze(1)
        size = len(data.t)
        if not (len(data.S) == size and len(data.I) == size and len(data.R) == size):
            raise ValueError("All lists must be of equal length.")
            
        y_data = torch.tensor(np.vstack([data.S, data.I, data.R]).T, dtype=torch.float32)
        
        t_max = max(data.t)
        t_physics = torch.linspace(0, t_max, int(t_max * 2)).unsqueeze(1)
        
        torch.manual_seed(42)
        model = SIR_PINN(hidden_layers=3, nodes=32)
        
        train_pinn(model, t_data, y_data, t_physics, epochs_adam=data.epochs_adam, epochs_lbfgs=data.epochs_lbfgs, verbose=False)
        
        beta_val = float(model.beta.item())
        gamma_val = float(model.gamma.item())
        
        if math.isnan(beta_val) or math.isnan(gamma_val):
            raise ValueError("Training diverged to NaN. The input dataset may be physically impossible to fit.")
            
        return {
            "status": "success",
            "learned_beta": beta_val,
            "learned_gamma": gamma_val
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/infer_from_csv")
async def infer_from_csv(file: UploadFile = File(...), epochs_adam: int = 500, epochs_lbfgs: int = 20):
    """
    Solves the Phase 3 specifically requested feature allowing dataset uploads directly via CSV.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        required_cols = {'t', 'S', 'I', 'R'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
            
        t_data = torch.tensor(df['t'].values, dtype=torch.float32).unsqueeze(1)
        y_data = torch.tensor(np.vstack([df['S'], df['I'], df['R']]).T, dtype=torch.float32)
        
        t_max = df['t'].max()
        t_physics = torch.linspace(0, t_max, int(t_max * 2)).unsqueeze(1)
        
        torch.manual_seed(42)
        model = SIR_PINN(hidden_layers=3, nodes=32)
        
        train_pinn(model, t_data, y_data, t_physics, epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs, verbose=False)
        
        beta_val = float(model.beta.item())
        gamma_val = float(model.gamma.item())
        
        if math.isnan(beta_val) or math.isnan(gamma_val):
            raise ValueError("Training diverged to NaN.")
            
        return {
            "status": "success",
            "learned_beta": beta_val,
            "learned_gamma": gamma_val,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
