# Linear Features Probe Experiments

This repository contains experiments to measure how strongly individual Sparse Autoencoder (SAE) features correspond to lexical or contextual information in transformer language models.

## Installation

### Base Dependencies (CPU/GPU)
```bash
uv pip install -r requirements.txt
```

### Optional: GPU-Accelerated Probes (NVIDIA GPUs only)
For significantly faster probe training on systems with NVIDIA GPUs and CUDA 12.x:
```bash
uv pip install -r requirements-gpu.txt
```

**Note**: `cuml` requires CUDA and will not work on Mac/CPU-only systems. The code will automatically fall back to PyTorch/sklearn if cuML is not available.

## Quick Start

### Benchmark Probe Training Backends
Test different probe training implementations to find the fastest option for your hardware:
```bash
python test_probe_backends.py
```

This will compare:
- PyTorch (current implementation)
- sklearn (CPU multi-threaded)
- cuML (GPU-accelerated, if available)

### Run Full Experiments
```bash
# Gemma-2-2B experiments
python run_probes_gemma2.py

# Gemma-2-9B experiments  
python run_probes_gemma2_9b.py

# Other models
python run_probes.py
python run_probes_gemma3.py
python run_probes_llama3.py
```

## Methodology

See [exp.md](exp.md) for detailed experimental design documentation.
