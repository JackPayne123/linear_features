"""
Lightweight comparison of probe training backends:
- Current PyTorch implementation
- sklearn LogisticRegression
- cuML LogisticRegression (if available)

This script trains on a small subset to quickly validate performance/accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import time

# Try importing cuML (GPU-only, optional)
try:
    from cuml.linear_model import LogisticRegression as cuMLLogisticRegression
    CUML_AVAILABLE = True
    print("✓ cuML is available (GPU acceleration enabled)")
except ImportError:
    CUML_AVAILABLE = False
    print("✗ cuML not available (optional - GPU only)")
    print("  To enable GPU acceleration on NVIDIA systems:")
    print("    uv pip install -r requirements-gpu.txt")

# Current PyTorch implementation (from your scripts)
class PyTorchLogisticRegression:
    def __init__(self, fit_intercept=True, max_iter=1000, device='cuda'):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.device = device
        self.model = None
        
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        input_dim = X.shape[1]
        
        if self.fit_intercept:
            self.model = nn.Linear(input_dim, 1).to(self.device)
        else:
            self.model = nn.Linear(input_dim, 1, bias=False).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        
        self.model.train()
        prev_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            optimizer.zero_grad()
            logits = self.model(X_tensor).squeeze()
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                current_loss = loss.item()
                if abs(prev_loss - current_loss) < 1e-6:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                else:
                    patience_counter = 0
                prev_loss = current_loss
            
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
        
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def generate_synthetic_data(n_samples=50000, n_features=2304, sparsity=0.1):
    """Generate synthetic data similar to your probe task"""
    np.random.seed(42)
    
    # Generate features (similar to embedding dimension)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels with some structure (not purely random)
    # Simulate sparse feature activation pattern
    true_weights = np.random.randn(n_features) * 0.1
    true_weights[np.random.rand(n_features) > sparsity] = 0
    
    logits = X @ true_weights + np.random.randn(n_samples) * 0.5
    y = (logits > np.median(logits)).astype(int)
    
    # Balance the training data
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    neg_sample = np.random.choice(neg_idx, size=n_pos, replace=False)
    balanced_idx = np.concatenate([pos_idx, neg_sample])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    # Create test set (80/20 split, unbalanced)
    split_idx = int(0.8 * n_samples)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_balanced, y_balanced, X_test, y_test


def benchmark_backend(name, model, X_train, y_train, X_test, y_test, device='cuda'):
    """Train and evaluate a model, returning timing and metrics"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start
    
    auc = roc_auc_score(y_test, probs)
    
    print(f"  Training time:   {train_time:.3f}s")
    print(f"  Inference time:  {inference_time:.3f}s")
    print(f"  AUC:             {auc:.4f}")
    
    return {
        'name': name,
        'train_time': train_time,
        'inference_time': inference_time,
        'auc': auc
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic data similar to your use case
    print("\nGenerating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=50000,
        n_features=2304,  # Gemma-2-2b d_model
        sparsity=0.1
    )
    
    print(f"  Train samples: {len(X_train)} (balanced)")
    print(f"  Test samples:  {len(X_test)} (natural distribution)")
    print(f"  Feature dim:   {X_train.shape[1]}")
    print(f"  Positive rate (test): {y_test.mean():.2%}")
    
    results = []
    
    # 1. Current PyTorch implementation
    pytorch_model = PyTorchLogisticRegression(
        fit_intercept=True,
        max_iter=500,
        device=device
    )
    results.append(benchmark_backend(
        "PyTorch (Current)",
        pytorch_model,
        X_train, y_train, X_test, y_test
    ))
    
    # 2. sklearn baseline
    sklearn_model = SklearnLogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1
    )
    results.append(benchmark_backend(
        "sklearn (CPU)",
        sklearn_model,
        X_train, y_train, X_test, y_test
    ))
    
    # 3. cuML if available
    if CUML_AVAILABLE:
        cuml_model = cuMLLogisticRegression(
            max_iter=100,
            tol=1e-4
        )
        results.append(benchmark_backend(
            "cuML (GPU)",
            cuml_model,
            X_train, y_train, X_test, y_test
        ))
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Backend':<20} {'Train Time':<12} {'Inference':<12} {'AUC':<8} {'Speedup':<8}")
    print("-" * 70)
    
    baseline_train_time = results[0]['train_time']
    baseline_auc = results[0]['auc']
    
    for r in results:
        speedup = baseline_train_time / r['train_time']
        auc_diff = r['auc'] - baseline_auc
        print(f"{r['name']:<20} {r['train_time']:>8.3f}s    {r['inference_time']:>8.3f}s    "
              f"{r['auc']:.4f}   {speedup:.2f}x")
    
    print("\nInterpretation:")
    print("  - AUC differences < 0.01 indicate equivalent model quality")
    print("  - Speedup > 5x would significantly reduce experiment time")
    print("  - cuML should be fastest for large-scale experiments")
    
    if CUML_AVAILABLE:
        cuml_result = [r for r in results if 'cuML' in r['name']][0]
        pytorch_result = results[0]
        
        speedup = pytorch_result['train_time'] / cuml_result['train_time']
        auc_diff = abs(cuml_result['auc'] - pytorch_result['auc'])
        
        print(f"\ncuML vs PyTorch:")
        print(f"  Training speedup: {speedup:.2f}x")
        print(f"  AUC difference:   {auc_diff:.4f}")
        
        if speedup > 2 and auc_diff < 0.01:
            print("  ✓ cuML is faster and produces equivalent results!")
            print("  Recommendation: Use cuML for full experiments")
        elif auc_diff >= 0.01:
            print("  ⚠ AUC difference is significant - investigate convergence")
        else:
            print("  → Speedup is marginal - current implementation may be fine")


if __name__ == "__main__":
    main()

