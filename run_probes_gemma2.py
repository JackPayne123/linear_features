import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import einops
import os

# Set CUDA memory allocation config to avoid fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
MODEL_NAME = "gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res"
# Using smaller 1k width SAE to reduce memory usage on 3060
# Available widths: width_1k, width_16k, width_65k, width_262k, width_1m
SAE_ID = "layer_20/width_16k/average_l0_71"  # Requested specific SAE
# Layer 0 hook will be determined from model structure
LAYER_0_HOOK = None  # Will be set dynamically
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

import pandas as pd

# PyTorch Logistic Regression for GPU acceleration
class PyTorchLogisticRegression:
    def __init__(self, fit_intercept=True, max_iter=1000, device='cuda'):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.device = device
        self.model = None
        
    def fit(self, X, y):
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        input_dim = X.shape[1]
        
        # Create model
        if self.fit_intercept:
            self.model = nn.Linear(input_dim, 1).to(self.device)
        else:
            self.model = nn.Linear(input_dim, 1, bias=False).to(self.device)
        
        # Training setup - use Adam for faster convergence on GPU
        # Add weight_decay=1e-4 to mimic L2 regularization (prevents overfitting on sparse features)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # Training loop with early stopping
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
            
            # Early stopping
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
        
        # Return in sklearn format: [[1-p, p], ...]
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

# Features to probe
FEATURES = {
    # 29007: "This detector",
    # 29004: "Medical/Physiological",
    # 27861: "Possessive pronouns",
    # 17939: "Healthcare/Medical treatments",
    # 22326: "negative descriptors associated with inappropriate behavior",
    249: "expressions related to financial fraud and ethical considerations in investment practices"
}

# Add features 500-600
for i in range(500, 601):
    if i not in FEATURES:
        FEATURES[i] = f"Feature {i}"

def get_device():
    return DEVICE

def load_model_and_sae():
    # Clear any existing cached models
    torch.cuda.empty_cache()
    
    print(f"Loading model: {MODEL_NAME}")
    # Use torch.float16 to save memory
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, 
        device=DEVICE,
        dtype=torch.float16  # Half precision to save memory
    )
    model.eval()  # Disable gradients
    
    print(f"Loading SAE: {SAE_RELEASE}/{SAE_ID}")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE
    )
    sae = sae.to(torch.float16)  # Convert SAE to half precision too
    sae.eval()  # Disable gradients
    
    # Print SAE config for debugging
    # Try to find hook name in config
    if hasattr(sae.cfg, "hook_name"):
        print(f"SAE hook point: {sae.cfg.hook_name}")
    elif hasattr(sae.cfg, "hook_point"):
        print(f"SAE hook point: {sae.cfg.hook_point}")
    else:
        print("Using manual hook derivation")
        
    print(f"SAE d_in: {sae.cfg.d_in}")
    print(f"SAE d_sae: {sae.cfg.d_sae}")
    
    # Report memory usage
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
    return model, sae

def get_embeddings(model, tokens):
    # tokens: [batch, seq]
    # Embeddings: W_E[tokens]
    
    W_E = model.W_E # [vocab, d_model]
    x_embed = W_E[tokens] # [batch, seq, d_model]
    
    # Gemma uses RoPE, so no W_pos at input
    # We will just reuse x_embed as x_embedpos to keep the loop structure valid
    x_embedpos = x_embed

    return x_embed, x_embedpos

import sklearn.linear_model

# ... (keep existing imports) ...

def run_experiment():
    # Set random seeds for reproducibility
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    model, sae = load_model_and_sae()
    
    # Determine layer 0 hook name from model
    layer_0_hook = "blocks.0.hook_resid_post"
    print(f"Using layer 0 hook: {layer_0_hook}")
    
    # ... (keep existing code) ...
    
    # Validate dimensions before running
    assert model.cfg.d_model == sae.cfg.d_in, \
        f"Model d_model ({model.cfg.d_model}) does not match SAE d_in ({sae.cfg.d_in})"
    
    print(f"Probing features: {FEATURES}")
    
    # Build Data
    print("Collecting sequences...")
    #dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    dataset = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    
    seq_embed = []
    seq_layer0 = [] # Store layer 0 activations
    seq_y = {k: [] for k in FEATURES}
    seq_tokens = [] # Store token IDs for analysis
    
    n_tokens = 5_000_000
    total_tokens = 0
    batch_size = 1 # Minimal batch size for Gemma 2 2B on RTX 3060
    
    iterator = iter(dataset)
    pbar = tqdm(total=n_tokens)
    
    # Determine hook name manually if not in config
    if hasattr(sae.cfg, "hook_name"):
        hook_name = sae.cfg.hook_name
    elif hasattr(sae.cfg, "hook_point"):
        hook_name = sae.cfg.hook_point
    else:
        # Hardcode fallback based on known SAE_ID structure
        # SAE_ID = "layer_20/..."
        import re
        match = re.search(r"layer_(\d+)", SAE_ID)
        if match:
            layer = int(match.group(1))
            hook_name = f"blocks.{layer}.hook_resid_post"
            print(f"Derived hook name from SAE_ID: {hook_name}")
        else:
            raise ValueError(f"Could not determine hook name from SAE config or ID. Config attrs: {dir(sae.cfg)}")
        
    print(f"\nUsing hook: {hook_name}")
    
    max_acts = {k: 0.0 for k in FEATURES}
    
    # Padding token handling
    pad_token_id = model.tokenizer.pad_token_id
    print(f"\nTokenizer Analysis:")
    print(f"  pad_token_id: {pad_token_id}")
    print(f"  Special tokens: {model.tokenizer.special_tokens_map}")
    
    if pad_token_id is None:
        print("\n✓ Tokenizer has no pad token. No padding filtering will be applied.")
        print("  Filtered vs Unfiltered experiments will be identical (no padding to filter).")
        should_filter_padding = False
    else:
        print(f"\n✓ Tokenizer has pad_token_id={pad_token_id}. Will filter padding tokens.")
        print("  Filtered experiment: removes pad tokens")
        print("  Unfiltered experiment: keeps pad tokens")
        print("  Comparison will reveal any padding-driven AUC inflation.")
        should_filter_padding = True

    while total_tokens < n_tokens:
        batch_texts = []
        while len(batch_texts) < batch_size:
            try:
                text = next(iterator)['text']
                if len(text) > 50: 
                    batch_texts.append(text)
            except StopIteration:
                break
        
        if not batch_texts: break
        
        tokens = model.to_tokens(batch_texts, truncate=True, prepend_bos=True)
        
        # Truncate to manageable sequence length
        if tokens.shape[1] > 1024: tokens = tokens[:, :1024]
        
        with torch.no_grad():
            x_embed, _ = get_embeddings(model, tokens)
            
            # Run with cache to get both layer 0 and SAE layer
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name, layer_0_hook])
            resid = cache[hook_name]
            layer0 = cache[layer_0_hook]
            
            # Dimensionality sanity check (only need to check once per batch technically, but cheap)
            assert resid.shape[-1] == sae.cfg.d_in, \
                f"Mismatch: Residual dim {resid.shape[-1]} != SAE d_in {sae.cfg.d_in}"
            
            feature_acts = sae.encode(resid)
            
            # Store activations for each feature
            for fid in FEATURES:
                y_feat = feature_acts[..., fid]
                current_max = y_feat.max().item()
                if current_max > max_acts[fid]:
                    max_acts[fid] = current_max
            
        # Append per sequence (iterate batch), filtering padding
        for i in range(tokens.shape[0]):
            # Optimization: Convert to numpy immediately to save GPU memory
            seq_embed.append(x_embed[i].cpu().numpy()) # [seq_len, d_model]
            seq_layer0.append(layer0[i].cpu().numpy()) # [seq_len, d_model]
            seq_tokens.append(tokens[i].cpu().numpy()) # [seq_len]
            
            for fid in FEATURES:
                seq_y[fid].append(feature_acts[..., fid][i].cpu().numpy()) # [seq_len]
                
            total_tokens += tokens.shape[1]
        
        # Clear GPU cache after each batch to prevent memory buildup
        del x_embed, resid, layer0, feature_acts, cache
        torch.cuda.empty_cache()
        
        pbar.update(tokens.shape[0] * tokens.shape[1])
        
    pbar.close()
    
    # Global Token Statistics
    all_tokens = np.concatenate(seq_tokens)
    print("\n" + "="*60)
    print("GLOBAL TOKEN STATISTICS")
    print("="*60)
    print(f"Total tokens collected: {len(all_tokens)}")
    unique_tokens = np.unique(all_tokens)
    print(f"Number of unique token types: {len(unique_tokens)}")
    print(f"Token ID range: [{unique_tokens.min()}, {unique_tokens.max()}]")
    print("First 100 unique tokens:", unique_tokens[:100].tolist())
    
    if should_filter_padding:
        pad_count = (all_tokens == pad_token_id).sum()
        print(f"\n{'PADDING ANALYSIS'}")
        print(f"  Pad token ID: {pad_token_id}")
        print(f"  Total pad tokens found: {pad_count} ({100*pad_count/len(all_tokens):.2f}%)")
        print(f"  Was pad token ever seen?: {pad_count > 0}")
        
        if pad_count > 0:
            print(f"\n  ⚠️  PADDING DETECTED: {pad_count} pad tokens found.")
            print(f"  This means the filtered vs unfiltered comparison is meaningful.")
            run_unfiltered = True
        else:
            print(f"\n  ✓ NO PADDING DETECTED: Filtered and unfiltered should be identical.")
            print("  Skipping unfiltered experiment to save time.")
            run_unfiltered = False
    else:
        print(f"\n{'NO PAD TOKEN DEFINED'}")
        print("  Checking for potential padding artifacts...")
        from collections import Counter
        counter = Counter(all_tokens.tolist())
        print("  Most common 10 tokens:", counter.most_common(10))
        print("\n  ✓ No padding filtering needed. Experiments will be identical.")
        run_unfiltered = False

    # Split sequences (indices only)
    indices = np.arange(len(seq_embed))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Define both filtered and unfiltered flatten functions
    def flatten_data_filtered(indices, s_embed, s_layer0, s_tokens, pad_id, should_filter):
        batch_X = []
        batch_L0 = []
        batch_tokens = []
        
        for i in indices:
            toks = s_tokens[i]
            # Create boolean mask: True if NOT padding
            if should_filter and pad_id is not None:
                mask = (toks != pad_id)
            else:
                mask = np.ones(len(toks), dtype=bool) # No filtering
            
            if mask.sum() > 0:
                batch_X.append(s_embed[i][mask])
                batch_L0.append(s_layer0[i][mask])
                batch_tokens.append(toks[mask])
                
        flat_X = np.concatenate(batch_X, axis=0)
        flat_L0 = np.concatenate(batch_L0, axis=0)
        flat_tokens = np.concatenate(batch_tokens, axis=0)
        
        return flat_X, flat_L0, flat_tokens
    
    def flatten_labels_filtered(indices, s_y_list, s_tokens, pad_id, should_filter):
        batch_y = []
        for i in indices:
            toks = s_tokens[i]
            if should_filter and pad_id is not None:
                mask = (toks != pad_id)
            else:
                mask = np.ones(len(toks), dtype=bool)
            if mask.sum() > 0:
                batch_y.append(s_y_list[i][mask])
        return np.concatenate(batch_y, axis=0)
    
    def flatten_data_unfiltered(indices, s_embed, s_layer0, s_tokens):
        flat_X = np.concatenate([s_embed[i] for i in indices], axis=0)
        flat_L0 = np.concatenate([s_layer0[i] for i in indices], axis=0)
        flat_tokens = np.concatenate([s_tokens[i] for i in indices], axis=0)
        return flat_X, flat_L0, flat_tokens

    def flatten_labels_unfiltered(indices, s_y_list):
        flat_y = np.concatenate([s_y_list[i] for i in indices], axis=0)
        return flat_y
    
    # Run BOTH experiments: filtered and unfiltered
    print("\n" + "="*60)
    print("EXPERIMENT 1: WITH PADDING FILTER")
    print("="*60)
    print("Flattening data arrays with padding filter...")
    
    X_train_filt, L0_train_filt, tokens_train_filt = flatten_data_filtered(train_idx, seq_embed, seq_layer0, seq_tokens, pad_token_id, should_filter_padding)
    X_test_filt, L0_test_filt, tokens_test_filt = flatten_data_filtered(test_idx, seq_embed, seq_layer0, seq_tokens, pad_token_id, should_filter_padding)
    
    # Count pads if they exist
    if should_filter_padding:
        all_train_tokens_unfilt = np.concatenate([seq_tokens[i] for i in train_idx])
        is_pad = (all_train_tokens_unfilt == pad_token_id)
        print(f"\nPadding Stats (before filtering):")
        print(f"  Total tokens in train: {len(all_train_tokens_unfilt)}")
        print(f"  Total pad tokens in train: {is_pad.sum()}")
        print(f"  After filtering: {len(tokens_train_filt)} tokens remain")
        print(f"  Tokens removed by filter: {len(all_train_tokens_unfilt) - len(tokens_train_filt)}")
    else:
        print(f"\nNo padding filtering applied.")
        print(f"  Total tokens in train: {len(tokens_train_filt)}")
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: WITHOUT PADDING FILTER (for comparison)")
    print("="*60)
    
    if run_unfiltered:
        print("Flattening data arrays WITHOUT padding filter...")
        X_train_unfilt, L0_train_unfilt, tokens_train_unfilt = flatten_data_unfiltered(train_idx, seq_embed, seq_layer0, seq_tokens)
        X_test_unfilt, L0_test_unfilt, tokens_test_unfilt = flatten_data_unfiltered(test_idx, seq_embed, seq_layer0, seq_tokens)
    else:
        print("Skipping Experiment 2 (Unfiltered) as no padding was found.")
        print("Unfiltered results will duplicate Filtered results for comparison consistency.")
        # Just point to filtered data to avoid crashes if referenced, though we won't train on them
        X_train_unfilt, L0_train_unfilt, tokens_train_unfilt = X_train_filt, L0_train_filt, tokens_train_filt
        X_test_unfilt, L0_test_unfilt, tokens_test_unfilt = X_test_filt, L0_test_filt, tokens_test_filt
    
    results_filtered = []
    results_unfiltered = []

    # Loop over each feature to train probes - BOTH versions
    print("\nTraining probes...")
    for fid, fname in tqdm(FEATURES.items(), desc="Training Probes"):
        
        max_act = max_acts[fid]
        
        # ==================== EXPERIMENT 1: FILTERED ====================
        y_train_raw_filt = flatten_labels_filtered(train_idx, seq_y[fid], seq_tokens, pad_token_id, should_filter_padding)
        y_test_raw_filt = flatten_labels_filtered(test_idx, seq_y[fid], seq_tokens, pad_token_id, should_filter_padding)
        
        # Dynamic Threshold Selection
        non_zero_acts = y_train_raw_filt[y_train_raw_filt > 0]
        if len(non_zero_acts) > 0:
            tau = 0.1 * max_act 
            threshold = tau
        else:
            threshold = 0.5
        
        y_train_filt = (y_train_raw_filt >= threshold).astype(int)
        y_test_filt = (y_test_raw_filt >= threshold).astype(int)
        
        # Check if test set has both classes - skip both experiments if not
        if len(np.unique(y_test_filt)) < 2:
            results_filtered.append({
                "feature_id": fid,
                "feature_name": fname,
                "probe_a_auc": np.nan,
                "probe_b_auc": np.nan,
                "probe_c_auc": np.nan,
                "n_pos_train": 0,
                "n_pos_test": (y_test_filt == 1).sum(),
                "n_total_train": 0,
                "n_total_test": len(y_test_filt)
            })
            if run_unfiltered:
                results_unfiltered.append({
                    "feature_id": fid,
                    "feature_name": fname,
                    "probe_a_auc": np.nan,
                    "probe_b_auc": np.nan,
                    "probe_c_auc": np.nan,
                    "n_pos_train": 0,
                    "n_pos_test": (y_test_filt == 1).sum(),
                    "n_total_train": 0,
                    "n_total_test": len(y_test_filt)
                })
            else:
                # Mirror the nan result
                results_unfiltered.append({
                    "feature_id": fid,
                    "feature_name": fname,
                    "probe_a_auc": np.nan,
                    "probe_b_auc": np.nan,
                    "probe_c_auc": np.nan,
                    "n_pos_train": 0,
                    "n_pos_test": (y_test_filt == 1).sum(),
                    "n_total_train": 0,
                    "n_total_test": len(y_test_filt)
                })
            continue
        
        # Balance Training Data
        pos_indices = np.where(y_train_filt == 1)[0]
        neg_indices = np.where(y_train_filt == 0)[0]
        
        n_pos = len(pos_indices)
        
        # Quality Check: Ensure enough positive examples
        n_pos_test_count = (y_test_filt == 1).sum()
        if n_pos < 30 or n_pos_test_count < 10:
            results_filtered.append({
                "feature_id": fid,
                "feature_name": fname,
                "probe_a_auc": np.nan,
                "probe_b_auc": np.nan,
                "probe_c_auc": np.nan,
                "n_pos_train": n_pos,
                "n_pos_test": n_pos_test_count,
                "n_total_train": 0,
                "n_total_test": len(y_test_filt)
            })
            if run_unfiltered:
                results_unfiltered.append({
                    "feature_id": fid,
                    "feature_name": fname,
                    "probe_a_auc": np.nan,
                    "probe_b_auc": np.nan,
                    "probe_c_auc": np.nan,
                    "n_pos_train": n_pos,
                    "n_pos_test": n_pos_test_count,
                    "n_total_train": 0,
                    "n_total_test": len(y_test_filt)
                })
            else:
                # Mirror the nan result
                results_unfiltered.append({
                    "feature_id": fid,
                    "feature_name": fname,
                    "probe_a_auc": np.nan,
                    "probe_b_auc": np.nan,
                    "probe_c_auc": np.nan,
                    "n_pos_train": n_pos,
                    "n_pos_test": n_pos_test_count,
                    "n_total_train": 0,
                    "n_total_test": len(y_test_filt)
                })
            continue

        neg_sample = np.random.choice(neg_indices, size=n_pos, replace=False)
        balanced_idx = np.concatenate([pos_indices, neg_sample])
        np.random.shuffle(balanced_idx)
        
        X_train_bal = X_train_filt[balanced_idx]
        L0_train_bal = L0_train_filt[balanced_idx]
        y_train_bal = y_train_filt[balanced_idx]
        
        # Train Probes (Filtered)
        probe_emb = PyTorchLogisticRegression(fit_intercept=True, max_iter=500, device=DEVICE)
        probe_emb.fit(X_train_bal, y_train_bal)
        probs_emb = probe_emb.predict_proba(X_test_filt)[:, 1]
        preds_emb = probe_emb.predict(X_test_filt)
        auc_emb_filt = roc_auc_score(y_test_filt, probs_emb)
        
        probe_l0 = PyTorchLogisticRegression(fit_intercept=True, max_iter=500, device=DEVICE)
        probe_l0.fit(L0_train_bal, y_train_bal)
        probs_l0 = probe_l0.predict_proba(L0_test_filt)[:, 1]
        preds_l0 = probe_l0.predict(L0_test_filt)
        auc_l0_filt = roc_auc_score(y_test_filt, probs_l0)
        
        probe_no_bias = PyTorchLogisticRegression(fit_intercept=False, max_iter=500, device=DEVICE)
        probe_no_bias.fit(X_train_bal, y_train_bal)
        probs_no_bias = probe_no_bias.predict_proba(X_test_filt)[:, 1]
        preds_no_bias = probe_no_bias.predict(X_test_filt)
        auc_no_bias_filt = roc_auc_score(y_test_filt, probs_no_bias)
        
        results_filtered.append({
            "feature_id": fid,
            "feature_name": fname,
            "probe_a_auc": auc_emb_filt,
            "probe_b_auc": auc_l0_filt,
            "probe_c_auc": auc_no_bias_filt,
            "n_pos_train": len(y_train_bal) // 2,  # Balanced, so half are positive
            "n_pos_test": (y_test_filt == 1).sum(),
            "n_total_train": len(y_train_bal),
            "n_total_test": len(y_test_filt)
        })
        
        # ==================== EXPERIMENT 2: UNFILTERED ====================
        if not run_unfiltered:
            # Copy results from filtered experiment
            results_unfiltered.append({
                "feature_id": fid,
                "feature_name": fname,
                "probe_a_auc": auc_emb_filt,
                "probe_b_auc": auc_l0_filt,
                "probe_c_auc": auc_no_bias_filt,
                "n_pos_train": len(y_train_bal) // 2,
                "n_pos_test": (y_test_filt == 1).sum(),
                "n_total_train": len(y_train_bal),
                "n_total_test": len(y_test_filt)
            })
            continue

        y_train_raw_unfilt = flatten_labels_unfiltered(train_idx, seq_y[fid])
        y_test_raw_unfilt = flatten_labels_unfiltered(test_idx, seq_y[fid])
        
        y_train_unfilt = (y_train_raw_unfilt >= threshold).astype(int)
        y_test_unfilt = (y_test_raw_unfilt >= threshold).astype(int)
        
        # Balance Training Data
        pos_indices_u = np.where(y_train_unfilt == 1)[0]
        neg_indices_u = np.where(y_train_unfilt == 0)[0]
        
        n_pos_u = len(pos_indices_u)
        if n_pos_u == 0 or len(np.unique(y_test_unfilt)) < 2:
            results_unfiltered.append({
                "feature_id": fid,
                "feature_name": fname,
                "probe_a_auc": np.nan,
                "probe_b_auc": np.nan,
                "probe_c_auc": np.nan,
                "n_pos_train": 0,
                "n_pos_test": (y_test_unfilt == 1).sum(),
                "n_total_train": 0,
                "n_total_test": len(y_test_unfilt)
            })
            continue

        neg_sample_u = np.random.choice(neg_indices_u, size=n_pos_u, replace=False)
        balanced_idx_u = np.concatenate([pos_indices_u, neg_sample_u])
        np.random.shuffle(balanced_idx_u)
        
        X_train_bal_u = X_train_unfilt[balanced_idx_u]
        L0_train_bal_u = L0_train_unfilt[balanced_idx_u]
        y_train_bal_u = y_train_unfilt[balanced_idx_u]
        
        # Train Probes (Unfiltered)
        probe_emb_u = PyTorchLogisticRegression(fit_intercept=True, max_iter=500, device=DEVICE)
        probe_emb_u.fit(X_train_bal_u, y_train_bal_u)
        probs_emb_u = probe_emb_u.predict_proba(X_test_unfilt)[:, 1]
        auc_emb_unfilt = roc_auc_score(y_test_unfilt, probs_emb_u)
        
        probe_l0_u = PyTorchLogisticRegression(fit_intercept=True, max_iter=500, device=DEVICE)
        probe_l0_u.fit(L0_train_bal_u, y_train_bal_u)
        probs_l0_u = probe_l0_u.predict_proba(L0_test_unfilt)[:, 1]
        auc_l0_unfilt = roc_auc_score(y_test_unfilt, probs_l0_u)
        
        probe_no_bias_u = PyTorchLogisticRegression(fit_intercept=False, max_iter=500, device=DEVICE)
        probe_no_bias_u.fit(X_train_bal_u, y_train_bal_u)
        probs_no_bias_u = probe_no_bias_u.predict_proba(X_test_unfilt)[:, 1]
        auc_no_bias_unfilt = roc_auc_score(y_test_unfilt, probs_no_bias_u)
        
        results_unfiltered.append({
            "feature_id": fid,
            "feature_name": fname,
            "probe_a_auc": auc_emb_unfilt,
            "probe_b_auc": auc_l0_unfilt,
            "probe_c_auc": auc_no_bias_unfilt,
            "n_pos_train": len(y_train_bal_u) // 2,
            "n_pos_test": (y_test_unfilt == 1).sum(),
            "n_total_train": len(y_train_bal_u),
            "n_total_test": len(y_test_unfilt)
        })

    # Save and Compare Results
    df_filt = pd.DataFrame(results_filtered)
    df_unfilt = pd.DataFrame(results_unfiltered)
    
    df_filt.to_csv("probe_results_gemma2_filtered.csv", index=False)
    df_unfilt.to_csv("probe_results_gemma2_unfiltered.csv", index=False)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON: FILTERED vs UNFILTERED")
    print("="*60)
    
    # Compute differences
    df_comparison = df_filt.copy()
    df_comparison["probe_a_auc_diff"] = df_unfilt["probe_a_auc"] - df_filt["probe_a_auc"]
    df_comparison["probe_b_auc_diff"] = df_unfilt["probe_b_auc"] - df_filt["probe_b_auc"]
    df_comparison["probe_c_auc_diff"] = df_unfilt["probe_c_auc"] - df_filt["probe_c_auc"]
    
    df_comparison.to_csv("probe_results_gemma2_comparison.csv", index=False)
    
    print("\nFiltered results saved to: probe_results_gemma2_filtered.csv")
    print("Unfiltered results saved to: probe_results_gemma2_unfiltered.csv")
    print("Comparison saved to: probe_results_gemma2_comparison.csv")
    
    print("\nAUC Difference Statistics (Unfiltered - Filtered):")
    print(f"Valid features: {df_comparison['probe_a_auc_diff'].notna().sum()} / {len(df_comparison)}")
    print(f"Probe A (Embedding) - Mean diff: {df_comparison['probe_a_auc_diff'].mean():.4f}, Std: {df_comparison['probe_a_auc_diff'].std():.4f}")
    print(f"Probe B (Layer 0)   - Mean diff: {df_comparison['probe_b_auc_diff'].mean():.4f}, Std: {df_comparison['probe_b_auc_diff'].std():.4f}")
    print(f"Probe C (Non-Affine)- Mean diff: {df_comparison['probe_c_auc_diff'].mean():.4f}, Std: {df_comparison['probe_c_auc_diff'].std():.4f}")
    
    print("\nInterpretation:")
    print("If padding were inflating AUC, we would see LARGE POSITIVE differences (unfiltered >> filtered).")
    print("If differences are near zero or small, padding is NOT the driver of performance.")
    
    # Detailed comparison table for select features
    print("\n" + "="*60)
    print("DETAILED COMPARISON: Select Features")
    print("="*60)
    print(f"{'Feature ID':<12} {'Name':<40} {'Filt':<8} {'Unfilt':<8} {'Diff':<8} {'Pos(Tr)':<8}")
    print("-" * 90)
    
    # Show first 10 features plus any named features (only valid ones)
    valid_comparison = df_comparison[df_comparison['probe_a_auc'].notna()]
    named_features = [249]
    features_to_show = []
    for fid in named_features:
        if fid in valid_comparison['feature_id'].values:
            features_to_show.append(fid)
    
    # Add first few numeric features
    remaining = [fid for fid in valid_comparison['feature_id'].values[:10] if fid not in features_to_show]
    features_to_show.extend(remaining[:max(0, 10 - len(features_to_show))])
    
    for fid in features_to_show:
        row = df_comparison[df_comparison['feature_id'] == fid].iloc[0]
        name = row['feature_name'][:40]
        auc_filt = row['probe_a_auc']
        auc_unfilt = df_unfilt[df_unfilt['feature_id'] == fid].iloc[0]['probe_a_auc']
        diff = row['probe_a_auc_diff']
        pos_train = int(row['n_pos_train'])
        print(f"{fid:<12} {name:<40} {auc_filt:.4f}   {auc_unfilt:.4f}   {diff:+.4f}   {pos_train:<8}")
    
    # Summary Statistics for Filtered
    print("\n" + "="*60)
    print("SUMMARY STATISTICS: FILTERED EXPERIMENT")
    print("="*60)
    for probe_col in ["probe_a_auc", "probe_b_auc", "probe_c_auc"]:
        valid_data = df_filt[df_filt[probe_col].notna()]
        print(f"\n{probe_col} Distribution (valid: {len(valid_data)}/{len(df_filt)}):")
        print(f"  < 0.5: {len(valid_data[valid_data[probe_col] < 0.5])}")
        print(f"  0.5 - 0.6: {len(valid_data[(valid_data[probe_col] >= 0.5) & (valid_data[probe_col] < 0.6)])}")
        print(f"  0.6 - 0.7: {len(valid_data[(valid_data[probe_col] >= 0.6) & (valid_data[probe_col] < 0.7)])}")
        print(f"  0.7 - 0.8: {len(valid_data[(valid_data[probe_col] >= 0.7) & (valid_data[probe_col] < 0.8)])}")
        print(f"  0.8 - 0.9: {len(valid_data[(valid_data[probe_col] >= 0.8) & (valid_data[probe_col] < 0.9)])}")
        print(f"  >= 0.9: {len(valid_data[valid_data[probe_col] >= 0.9])}")
        
        print(f"  Low performing features (< 0.7):")
        low_perf = valid_data[valid_data[probe_col] < 0.7]
        for _, row in low_perf.head(10).iterrows():
            print(f"    Feature {row['feature_id']}: {row['feature_name']} (AUC: {row[probe_col]:.4f})")

    # ==================== SANITY CHECK: SKLEARN COMPARISON ====================
    print("\n" + "="*60)
    print("SANITY CHECK: Sklearn vs PyTorch (Top 5 Features)")
    print("="*60)
    
    # Select top 5 features by AUC
    top_features = df_filt.sort_values("probe_a_auc", ascending=False).head(5)
    
    print(f"{'Feature ID':<12} {'PyTorch AUC':<12} {'Sklearn AUC':<12} {'Diff':<8}")
    print("-" * 60)
    
    from sklearn.linear_model import LogisticRegression
    
    for _, row in top_features.iterrows():
        fid = row['feature_id']
        
        # Re-extract data for this feature (need to reproduce the exact split/balancing)
        # Note: This relies on seed=0 being set at start of run_experiment
        
        # Get raw data (filtered)
        y_train_raw = flatten_labels_filtered(train_idx, seq_y[fid], seq_tokens, pad_token_id, should_filter_padding)
        y_test_raw = flatten_labels_filtered(test_idx, seq_y[fid], seq_tokens, pad_token_id, should_filter_padding)
        
        # Thresholding
        max_act = max_acts[fid]
        non_zero_acts = y_train_raw[y_train_raw > 0]
        threshold = 0.1 * max_act if len(non_zero_acts) > 0 else 0.5
        
        y_train = (y_train_raw >= threshold).astype(int)
        y_test = (y_test_raw >= threshold).astype(int)
        
        # Balancing
        pos_indices = np.where(y_train == 1)[0]
        neg_indices = np.where(y_train == 0)[0]
        
        n_pos = len(pos_indices)
        if n_pos == 0: continue
            
        # Seed must match exactly what happened in the main loop
        # This is tricky because we called random.choice/shuffle many times.
        # To do this perfectly, we should have saved the balanced datasets.
        # BUT, for a "sanity check", just retraining on a NEW balanced split is fine 
        # as long as we compare PyTorch vs Sklearn on THAT new split.
        
        neg_sample = np.random.choice(neg_indices, size=n_pos, replace=False)
        balanced_idx = np.concatenate([pos_indices, neg_sample])
        np.random.shuffle(balanced_idx)
        
        X_train_bal = X_train_filt[balanced_idx]
        y_train_bal = y_train[balanced_idx]
        
        # Train PyTorch again on this specific split
        pt_model = PyTorchLogisticRegression(fit_intercept=True, max_iter=500, device=DEVICE)
        pt_model.fit(X_train_bal, y_train_bal)
        pt_probs = pt_model.predict_proba(X_test_filt)[:, 1]
        pt_auc = roc_auc_score(y_test, pt_probs)
        
        # Train Sklearn on CPU
        # Use same settings: C=1.0 (default), l2 penalty (default)
        sk_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', n_jobs=-1)
        sk_model.fit(X_train_bal, y_train_bal)
        sk_probs = sk_model.predict_proba(X_test_filt)[:, 1]
        sk_auc = roc_auc_score(y_test, sk_probs)
        
        diff = abs(pt_auc - sk_auc)
        print(f"{fid:<12} {pt_auc:.4f}       {sk_auc:.4f}       {diff:.4f}")
        
    print("\nInterpretation:")
    print("Differences < 0.01 indicate PyTorch implementation is correct.")
    print("Differences > 0.01 might be due to convergence tolerances or solver differences (Adam vs LBFGS).")

if __name__ == "__main__":
    run_experiment()



