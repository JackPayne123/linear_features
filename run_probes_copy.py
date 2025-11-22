import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import einops

# Configuration
MODEL_NAME = "pythia-70m-deduped"
SAE_RELEASE = "pythia-70m-deduped-res-sm" 
SAE_ID = "blocks.4.hook_resid_post"
LAYER_0_HOOK = "blocks.0.hook_resid_post"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

import pandas as pd

# Features to probe
FEATURES = {
    29007: "This detector",
    29004: "Medical/Physiological",
    27861: "Possessive pronouns",
    17939: "Healthcare/Medical treatments",
    22326: "negative descriptors associated with inappropriate behavior",
    249: "expressions related to financial fraud and ethical considerations in investment practices"
}

# Add features 500-700
for i in range(500, 701):
    if i not in FEATURES:
        FEATURES[i] = f"Feature {i}"

def get_device():
    return DEVICE

def load_model_and_sae():
    print(f"Loading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    
    print(f"Loading SAE: {SAE_RELEASE}/{SAE_ID}")
    try:
        sae = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=SAE_ID,
            device=DEVICE
        )
    except Exception as e:
        print(f"Failed to load specific SAE release {SAE_RELEASE}. Listing available...")
        # Fallback or error handling
        raise e
        
    return model, sae

def get_embeddings(model, tokens):
    # tokens: [batch, seq]
    # Embeddings: W_E[tokens]
    
    W_E = model.W_E # [vocab, d_model]
    x_embed = W_E[tokens] # [batch, seq, d_model]
    
    # POSITIONAL HANDLING FOR ROPE MODELS (Pythia)
    # Pythia uses RoPE, so no W_pos at input.
    # User instruction: Probe raw embeddings only.
    # We will just reuse x_embed as x_embedpos to keep the loop structure valid but redundant
    x_embedpos = x_embed

    return x_embed, x_embedpos

def run_experiment():
    model, sae = load_model_and_sae()
    
    print(f"Probing features: {FEATURES}")
    
    # Build Data
    print("Collecting sequences...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    seq_embed = []
    seq_layer0 = [] # Store layer 0 activations
    seq_y = {k: [] for k in FEATURES}
    seq_tokens = [] # Store token IDs for analysis
    
    n_tokens = 500_000
    total_tokens = 0
    batch_size = 32 # Increased batch size for speed
    
    iterator = iter(dataset)
    pbar = tqdm(total=n_tokens)
    
    # Determine hook name
    hook_name = getattr(sae.cfg, "hook_name", getattr(sae.cfg, "hook_point", SAE_ID))
    
    max_acts = {k: 0.0 for k in FEATURES}
    
    # Pre-allocate lists to avoid dynamic growth issues, though simple list append is usually fine for <1GB
    # We will just stick to lists but ensure we check memory if needed.
    # Fix for Padding: We need to identify padding. Pythia's pad token is usually 1.
    # We will filter tokens where token_id != pad_token_id
    
    pad_token_id = model.tokenizer.pad_token_id
    print(f"Tokenizer pad_token_id: {pad_token_id}")
    if pad_token_id is None:
        print("Tokenizer has no pad token. No padding should appear.")
        # We set it to -1 (impossible token) so the filter does nothing but is valid code
        pad_token_id = -1 

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
        
        # Rigorous Evidence Check 1: Batch Tokens
        unique_toks = torch.unique(tokens)
        # print("Unique token IDs in this batch (first 50):", unique_toks[:50].tolist())
        if pad_token_id != -1:
            print(f"Does there exist token_id == {pad_token_id}?:", (unique_toks == pad_token_id).any().item())
            
        if tokens.shape[1] > 128: tokens = tokens[:, :128]
        
        with torch.no_grad():
            x_embed, _ = get_embeddings(model, tokens)
            
            # Run with cache to get both layer 0 and SAE layer
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name, LAYER_0_HOOK])
            resid = cache[hook_name]
            layer0 = cache[LAYER_0_HOOK]
            
            feature_acts = sae.encode(resid)
            
            # Store activations for each feature
            for fid in FEATURES:
                y_feat = feature_acts[..., fid]
                current_max = y_feat.max().item()
                if current_max > max_acts[fid]:
                    max_acts[fid] = current_max
            
        # Append per sequence (iterate batch), filtering padding
        for i in range(tokens.shape[0]):
            # Create mask for non-padding tokens
            # Note: Pythia tokenizer often uses 1 as pad, but let's verify
            # The tokens tensor is on GPU, need to move to CPU for numpy ops or keep on GPU
            
            # We will perform filtering during the flattening stage to keep the collection loop fast
            # Just store everything for now, but we must be mindful of memory.
            
            # Optimization: Convert to numpy immediately to save GPU memory
            seq_embed.append(x_embed[i].cpu().numpy()) # [seq_len, d_model]
            seq_layer0.append(layer0[i].cpu().numpy()) # [seq_len, d_model]
            seq_tokens.append(tokens[i].cpu().numpy()) # [seq_len]
            
            for fid in FEATURES:
                seq_y[fid].append(feature_acts[..., fid][i].cpu().numpy()) # [seq_len]
                
            total_tokens += tokens.shape[1]
        
        pbar.update(tokens.shape[0] * tokens.shape[1])
        
    pbar.close()
    
    # Rigorous Evidence Check 2: Global Token Stats
    all_tokens = np.concatenate(seq_tokens)
    print("\nGlobal Token Stats:")
    print("Global unique tokens (first 200):", np.unique(all_tokens)[:200])
    if pad_token_id != -1:
        print(f"Was pad token_id {pad_token_id} ever seen?", (all_tokens == pad_token_id).any())
    else:
        print("Pad token is None. Checking for potential padding artifacts (e.g. frequent 0 or 1)...")
        from collections import Counter
        counter = Counter(all_tokens.tolist())
        print("Most common 10 tokens:", counter.most_common(10))

    # Split sequences (indices only)
    indices = np.arange(len(seq_embed))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Define both filtered and unfiltered flatten functions
    def flatten_data_filtered(indices, s_embed, s_layer0, s_tokens, pad_id):
        # We need to filter out padding tokens if pad_id is valid
        batch_X = []
        batch_L0 = []
        batch_tokens = []
        
        for i in indices:
            toks = s_tokens[i]
            # Create boolean mask: True if NOT padding
            if pad_id is not None and pad_id != -1:
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
    
    def flatten_labels_filtered(indices, s_y_list, s_tokens, pad_id):
        batch_y = []
        for i in indices:
            toks = s_tokens[i]
            if pad_id is not None and pad_id != -1:
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
    
    X_train_filt, L0_train_filt, tokens_train_filt = flatten_data_filtered(train_idx, seq_embed, seq_layer0, seq_tokens, pad_token_id)
    X_test_filt, L0_test_filt, tokens_test_filt = flatten_data_filtered(test_idx, seq_embed, seq_layer0, seq_tokens, pad_token_id)
    
    # Count pads if they exist
    if pad_token_id is not None and pad_token_id != -1:
        all_train_tokens_unfilt = np.concatenate([seq_tokens[i] for i in train_idx])
        is_pad = (all_train_tokens_unfilt == pad_token_id)
        print(f"\nPadding Stats (before filtering):")
        print(f"  Total tokens in train: {len(all_train_tokens_unfilt)}")
        print(f"  Total pad tokens in train: {is_pad.sum()}")
        print(f"  After filtering: {len(tokens_train_filt)} tokens remain")
        print(f"  Tokens removed by filter: {len(all_train_tokens_unfilt) - len(tokens_train_filt)}")
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: WITHOUT PADDING FILTER (for comparison)")
    print("="*60)
    print("Flattening data arrays WITHOUT padding filter...")
    
    X_train_unfilt, L0_train_unfilt, tokens_train_unfilt = flatten_data_unfiltered(train_idx, seq_embed, seq_layer0, seq_tokens)
    X_test_unfilt, L0_test_unfilt, tokens_test_unfilt = flatten_data_unfiltered(test_idx, seq_embed, seq_layer0, seq_tokens)
    
    results_filtered = []
    results_unfiltered = []

    # Loop over each feature to train probes - BOTH versions
    print("\nTraining probes for BOTH experiments...")
    for fid, fname in tqdm(FEATURES.items(), desc="Training Probes"):
        
        max_act = max_acts[fid]
        
        # ==================== EXPERIMENT 1: FILTERED ====================
        y_train_raw_filt = flatten_labels_filtered(train_idx, seq_y[fid], seq_tokens, pad_token_id)
        y_test_raw_filt = flatten_labels_filtered(test_idx, seq_y[fid], seq_tokens, pad_token_id)
        
        # Dynamic Threshold Selection
        non_zero_acts = y_train_raw_filt[y_train_raw_filt > 0]
        if len(non_zero_acts) > 0:
            tau = 0.1 * max_act 
            threshold = tau
        else:
            threshold = 0.5
        
        y_train_filt = (y_train_raw_filt >= threshold).astype(int)
        y_test_filt = (y_test_raw_filt >= threshold).astype(int)
        
        # Balance Training Data
        pos_indices = np.where(y_train_filt == 1)[0]
        neg_indices = np.where(y_train_filt == 0)[0]
        
        n_pos = len(pos_indices)
        if n_pos == 0:
            continue

        neg_sample = np.random.choice(neg_indices, size=n_pos, replace=False)
        balanced_idx = np.concatenate([pos_indices, neg_sample])
        np.random.shuffle(balanced_idx)
        
        X_train_bal = X_train_filt[balanced_idx]
        L0_train_bal = L0_train_filt[balanced_idx]
        y_train_bal = y_train_filt[balanced_idx]
        
        # Train Probes (Filtered)
        probe_emb = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
        probe_emb.fit(X_train_bal, y_train_bal)
        probs_emb = probe_emb.predict_proba(X_test_filt)[:, 1]
        preds_emb = probe_emb.predict(X_test_filt)
        auc_emb_filt = roc_auc_score(y_test_filt, probs_emb)
        
        probe_l0 = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
        probe_l0.fit(L0_train_bal, y_train_bal)
        probs_l0 = probe_l0.predict_proba(L0_test_filt)[:, 1]
        preds_l0 = probe_l0.predict(L0_test_filt)
        auc_l0_filt = roc_auc_score(y_test_filt, probs_l0)
        
        probe_no_bias = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=False)
        probe_no_bias.fit(X_train_bal, y_train_bal)
        probs_no_bias = probe_no_bias.predict_proba(X_test_filt)[:, 1]
        preds_no_bias = probe_no_bias.predict(X_test_filt)
        auc_no_bias_filt = roc_auc_score(y_test_filt, probs_no_bias)
        
        results_filtered.append({
            "feature_id": fid,
            "feature_name": fname,
            "probe_a_auc": auc_emb_filt,
            "probe_b_auc": auc_l0_filt,
            "probe_c_auc": auc_no_bias_filt
        })
        
        # ==================== EXPERIMENT 2: UNFILTERED ====================
        y_train_raw_unfilt = flatten_labels_unfiltered(train_idx, seq_y[fid])
        y_test_raw_unfilt = flatten_labels_unfiltered(test_idx, seq_y[fid])
        
        y_train_unfilt = (y_train_raw_unfilt >= threshold).astype(int)
        y_test_unfilt = (y_test_raw_unfilt >= threshold).astype(int)
        
        # Balance Training Data
        pos_indices_u = np.where(y_train_unfilt == 1)[0]
        neg_indices_u = np.where(y_train_unfilt == 0)[0]
        
        n_pos_u = len(pos_indices_u)
        if n_pos_u == 0:
            results_unfiltered.append({
                "feature_id": fid,
                "feature_name": fname,
                "probe_a_auc": np.nan,
                "probe_b_auc": np.nan,
                "probe_c_auc": np.nan
            })
            continue

        neg_sample_u = np.random.choice(neg_indices_u, size=n_pos_u, replace=False)
        balanced_idx_u = np.concatenate([pos_indices_u, neg_sample_u])
        np.random.shuffle(balanced_idx_u)
        
        X_train_bal_u = X_train_unfilt[balanced_idx_u]
        L0_train_bal_u = L0_train_unfilt[balanced_idx_u]
        y_train_bal_u = y_train_unfilt[balanced_idx_u]
        
        # Train Probes (Unfiltered)
        probe_emb_u = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
        probe_emb_u.fit(X_train_bal_u, y_train_bal_u)
        probs_emb_u = probe_emb_u.predict_proba(X_test_unfilt)[:, 1]
        auc_emb_unfilt = roc_auc_score(y_test_unfilt, probs_emb_u)
        
        probe_l0_u = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
        probe_l0_u.fit(L0_train_bal_u, y_train_bal_u)
        probs_l0_u = probe_l0_u.predict_proba(L0_test_unfilt)[:, 1]
        auc_l0_unfilt = roc_auc_score(y_test_unfilt, probs_l0_u)
        
        probe_no_bias_u = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=False)
        probe_no_bias_u.fit(X_train_bal_u, y_train_bal_u)
        probs_no_bias_u = probe_no_bias_u.predict_proba(X_test_unfilt)[:, 1]
        auc_no_bias_unfilt = roc_auc_score(y_test_unfilt, probs_no_bias_u)
        
        results_unfiltered.append({
            "feature_id": fid,
            "feature_name": fname,
            "probe_a_auc": auc_emb_unfilt,
            "probe_b_auc": auc_l0_unfilt,
            "probe_c_auc": auc_no_bias_unfilt
        })

    # Save and Compare Results
    df_filt = pd.DataFrame(results_filtered)
    df_unfilt = pd.DataFrame(results_unfiltered)
    
    df_filt.to_csv("probe_results_filtered.csv", index=False)
    df_unfilt.to_csv("probe_results_unfiltered.csv", index=False)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON: FILTERED vs UNFILTERED")
    print("="*60)
    
    # Compute differences
    df_comparison = df_filt.copy()
    df_comparison["probe_a_auc_diff"] = df_unfilt["probe_a_auc"] - df_filt["probe_a_auc"]
    df_comparison["probe_b_auc_diff"] = df_unfilt["probe_b_auc"] - df_filt["probe_b_auc"]
    df_comparison["probe_c_auc_diff"] = df_unfilt["probe_c_auc"] - df_filt["probe_c_auc"]
    
    df_comparison.to_csv("probe_results_comparison.csv", index=False)
    
    print("\nFiltered results saved to: probe_results_filtered.csv")
    print("Unfiltered results saved to: probe_results_unfiltered.csv")
    print("Comparison saved to: probe_results_comparison.csv")
    
    print("\nAUC Difference Statistics (Unfiltered - Filtered):")
    print(f"Probe A (Embedding) - Mean diff: {df_comparison['probe_a_auc_diff'].mean():.4f}, Std: {df_comparison['probe_a_auc_diff'].std():.4f}")
    print(f"Probe B (Layer 0)   - Mean diff: {df_comparison['probe_b_auc_diff'].mean():.4f}, Std: {df_comparison['probe_b_auc_diff'].std():.4f}")
    print(f"Probe C (Non-Affine)- Mean diff: {df_comparison['probe_c_auc_diff'].mean():.4f}, Std: {df_comparison['probe_c_auc_diff'].std():.4f}")
    
    print("\nInterpretation:")
    print("If padding were inflating AUC, we would see LARGE POSITIVE differences (unfiltered >> filtered).")
    print("If differences are near zero or small, padding is NOT the driver of performance.")
    
    # Summary Statistics for Filtered
    print("\n" + "="*60)
    print("SUMMARY STATISTICS: FILTERED EXPERIMENT")
    print("="*60)
    for probe_col in ["probe_a_auc", "probe_b_auc", "probe_c_auc"]:
        print(f"\n{probe_col} Distribution:")
        print(f"  < 0.5: {len(df_filt[df_filt[probe_col] < 0.5])}")
        print(f"  0.5 - 0.6: {len(df_filt[(df_filt[probe_col] >= 0.5) & (df_filt[probe_col] < 0.6)])}")
        print(f"  0.6 - 0.7: {len(df_filt[(df_filt[probe_col] >= 0.6) & (df_filt[probe_col] < 0.7)])}")
        print(f"  0.7 - 0.8: {len(df_filt[(df_filt[probe_col] >= 0.7) & (df_filt[probe_col] < 0.8)])}")
        print(f"  0.8 - 0.9: {len(df_filt[(df_filt[probe_col] >= 0.8) & (df_filt[probe_col] < 0.9)])}")
        print(f"  >= 0.9: {len(df_filt[df_filt[probe_col] >= 0.9])}")
        
        print(f"  Low performing features (< 0.7):")
        low_perf = df_filt[df_filt[probe_col] < 0.7]
        for _, row in low_perf.iterrows():
            print(f"    Feature {row['feature_id']}: {row['feature_name']} (AUC: {row[probe_col]:.4f})")

if __name__ == "__main__":
    model, sae = load_model_and_sae()
    print(model.tokenizer.convert_ids_to_tokens([0]))