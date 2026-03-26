# 🦠 SIRA1: Physics-Informed Neural Network (PINN) Epidemic Tracker

This repository implements a robust Physics-Informed Neural Network (PINN) using PyTorch's Automatic Differentiation to natively map $SIR$ and $SEIR$ dynamical Ordinary Differential Equation (ODE) constraints onto real and synthetic epidemic datasets. 

🚧 **Under Active GSoC Development** 🚧

## 🌟 Core Highlights
Traditional neural networks often fail on highly sparse or noisy epidemic data. By encoding mathematical compartmental models directly into the loss function, the network learns the hidden infection rates ($\beta$) and recovery rates ($\gamma$) flawlessly, even with high noise thresholds.

1. **Dual-Stage Optimization Framework**: Leverages rapid gradient mappings via `Adam` pre-training gracefully transitioning into highly precise `L-BFGS` fine-tuning.
2. **Hard Initial Condition Anchoring**: Enforces strict starting states ($S_0, I_0, R_0$) structurally bypassing gradient collapse points.
3. **Adaptive Log-Variance Weighting**: Features completely dynamic, self-balancing loss components (`w_data`, `w_physics`) via log-variance optimizations resolving challenging inverse problems mathematically.
4. **Data Agnostic Pipeline**: Switches natively between generating high-entropy synthetic data (with Gaussian noise generators) or pulling active real-world statistics seamlessly from the JHU CSSE COVID-19 repository.

---

## 📂 Project Structure
```text
SIRA1/
├── api/
│   ├── main.py        # FastAPI Endpoints for HTTP parameter inference
│   └── test_api.py    # Python CLI API interaction testing tool
├── core/
│   ├── model.py       # PINN architecture, Parameter hooks, Autograd ODE physics
│   ├── solver.py      # SciPy numerical baseline integrators
│   ├── cli.py         # Argparse command line training pipeline
│   └── trainer.py     # Main Adam & L-BFGS multi-stage iterative loops
├── dashboard/
│   └── app.py         # Real-time Streamlit data visualizer UI
├── data/
│   ├── generator.py   # ODE Gaussian synthetic parameter generator
│   └── ingestion.py   # JHU CSSE API fetchers and normalized fraction hooks
└── notebooks/
    └── historical.py  # Validation pipelines tracking the 1918 Flu parameters
```

---

## 🚀 Getting Started

### 1. Installation Environment
Clone the repository and set up your Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Streamlit Visual Dashboard (Recommended)
You can visually configure noise boundaries and epochs in real-time, instantly visualizing the PINN's learned curves.
```bash
streamlit run dashboard/app.py
```
> **Tip:** Adjust the Adam Epochs to >2000 iteratively for high noise scenarios!

### 3. Automated Command Line Benchmarks
For researchers running strict computational parameter tests seamlessly via CI/CD.
```bash
# Run the synthetic ODE generator matching 5% noise!
python core/cli.py --noise 0.05 --epochs-adam 2000 --epochs-lbfgs 20
```

### 4. Headless Model Inference via FastAPI
Serve the model architecture to external downstream applications natively.
```bash
# Terminal 1: Launch FastAPI Endpoint
fastapi dev api/main.py

# Terminal 2: Test JSON inference
python api/test_api.py
```
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test inputs via Swagger UI!

---
## 🧪 Testing Methodology
All core algorithmic mathematical differentials and initial states are validated rigorously via `pytest`:
```bash
pytest tests/
```
