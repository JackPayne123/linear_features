
# Experimental Design and Methodology Overview

## 1. Objective

The purpose of this experimental framework is to measure how strongly individual Sparse Autoencoder (SAE) features correspond to lexical or contextual information encoded in a transformer language model’s internal representations. The experiment evaluates whether a given SAE feature can be predicted using linear probes trained on:

1. The model’s **input token embeddings** (lexical baseline).
2. The model’s **first layer residual stream** (early contextual baseline).
3. A **zero-bias linear probe on embeddings** (additional linear baseline).

This allows us to quantify the degree to which an SAE feature reflects:

* purely token identity,
* emergent contextual structure,
* deeper nonlinear model behaviour.

The pipeline is model-agnostic and has been tested on architectures ranging from **Pythia-70M** to **Gemma-2B**, with the intention to scale further.

---

## 2. High-Level Procedure

The experiment proceeds in three major stages:

1. **Collect activations from a pretrained transformer**

   * Sample approximately 500k tokens from a standard natural text corpus (Wikitext-103).
   * Record token embeddings and early-layer residual activations.
   * Record corresponding SAE feature activations at a chosen layer.

2. **Construct probe datasets**

   * Convert continuous SAE activations into binary labels using a dynamic per-feature threshold.
   * Split into training and test sequences.
   * Balance the training dataset (equal positives and negatives).
   * Leave the test dataset unbalanced to reflect the natural distribution.

3. **Train and evaluate linear probes**

   * Train logistic regression models (embedding, layer 0 residual, embedding no-bias).
   * Evaluate on the natural test distribution using AUC, balanced accuracy, etc.
   * (Optional) Compare filtered vs unfiltered runs to detect padding-based confounds.

The driver now loops over a small list of SAE configurations (`SAE_CONFIGS`). Each entry specifies the release/id pair and a short `name` (e.g., `layer20_width16k_l0_71`, `layer25_width16k_l0_116`, `layer12_width16k_l0_82`). The entire collection/flatten/probing pipeline is rerun for every config so that layer-specific results stay isolated.

This yields a quantitative estimate of how well a probe can predict SAE feature activation from shallow model representations.

---

## 3. Dataset Construction

### 3.1 Source Data

* Text: streaming Wikitext-103 (train split only).
* No document-level structure retained.
* Approximately 500k tokens collected.

### 3.2 Tokenisation

* Use the tokenizer associated with each model.
* Record:

  * token IDs,
  * embedding vectors,
  * residual stream activations at layer 0,
  * SAE encoder activations at the selected layer.

### 3.3 Padding Handling

Different models handle padding differently.
To guarantee the results are not padded-token artefacts, the pipeline performs:

* **Filtered experiment**
  Remove any tokens whose ID equals the model’s `pad_token_id`.

* **Unfiltered experiment**
  Keep all tokens.

* **Comparison**
  Compute per-feature AUC differences between filtered and unfiltered runs.
  If padding drives performance, unfiltered AUC should be dramatically higher.

This enables universal portability across models with:

* no pad token (e.g. Pythia),
* explicit pad token (e.g. Gemma),
* token ID zero representing meaningful tokens vs padding.

---

## 4. Feature Labelling Strategy

### 4.1 Collect Continuous SAE Activations

For each feature index f:

* Extract its activation value at each token position.

### 4.2 Dynamic Thresholding

Instead of using a fixed threshold for all features, define:

```
threshold_f = 0.10 × max_activation_f
```

where `max_activation_f` is the maximum observed activation for feature f in the training set.

This yields a per-feature binary target:

```
label = 1  if  activation >= threshold_f
         0  otherwise
```

Thresholding ensures the probe is learning genuine feature-specific structure rather than noise.

---

## 5. Train/Test Construction

### 5.1 Train/Test Split

* Split by sequence index (not by token) to avoid leakage.
* 80 percent train, 20 percent test.

### 5.2 Balanced Training Set

For each feature:

* Identify all positive examples in train.
* Randomly sample an equal number of negative examples.
* Concatenate and shuffle.

Reason:

* Prevent trivial majority-class solutions.
* Ensure the probe must learn meaningful geometry.

### 5.3 Natural Test Distribution

No balancing is applied to the test set.
This ensures AUC, precision, and recall reflect real-world prevalence.

---

## 6. Linear Probes

### 6.1 Probes Trained

For each feature:

1. **Probe A: Token Embeddings + Bias**
   Tests whether the feature is primarily lexical.

2. **Probe B: Residual Stream at Layer 0 + Bias**
   Tests whether early contextual information improves prediction.

3. **Probe C: Token Embeddings Without Bias**
   Tests whether a purely linear hyperplane (no intercept) suffices.

### 6.2 Implementation

Two backends supported:

* **Sklearn Logistic Regression** (for smaller models)
* **Custom PyTorch Logistic Regression**

  * Supports GPU acceleration
  * Supports half-precision for large models like Gemma-2B
  * Includes early stopping
  * Equivalent functional behaviour to sklearn

### 6.3 Evaluation Metrics

Main metric: **ROC AUC**

* insensitive to class imbalance
* well-behaved on natural test distribution
* directly comparable across features

Supplementary metrics:

* balanced accuracy
* raw accuracy
* precision
* recall

AUC is the key output because it is threshold-free and model-agnostic.

---

## 7. Padding-Confound Analysis

To rule out padding artefacts:

1. **Run filtered experiment**

   * Drop all pad tokens from X and Y.

2. **Run unfiltered experiment**

   * Include pad tokens.

3. **Compute delta AUC**

```
Δ_f = AUC_unfiltered_f - AUC_filtered_f
```

Interpretation:

* If Δ_f ≈ 0, padding is not a confound.
* If Δ_f is large and positive, the model is accidentally learning pad-token geometry.

This produces a rigorous model-agnostic confound check.

---

## 8. Cross-Model Generalisation

The pipeline is designed to be modular.
To test a new model, you only need to specify:

* model name,
* SAE release name,
* SAE feature index subset,
* target layer for SAE activations.

The same procedure has already been used for:

* **Pythia-70M**
* **Gemma-2B**

and can be extended to any RoPE-based or absolute-position model.

No assumptions are made about the embedding matrix, tokenizer format, or padding strategy.

---

## 9. Outputs

For each model and each feature, the experiment produces:

1. Per-feature metrics (for filtered and unfiltered conditions):

   * Probe A/B/C AUC
   * Positive counts in train/test
   * Total train/test sizes

2. Comparison tables:

   * Filtered vs unfiltered AUC differences per probe
   * Aggregated statistics (mean Δ, std Δ)

3. Summary distributions:

   * histograms of AUCs
   * list of poorly predicted features
   * optional visualisations (UMAP, scatter plots)

4. Per-feature diagnostics for qualitative follow-up:

   * `per_feature_analysis_gemma2.json` captures, for every probed feature, the dynamic threshold that was chosen, token-level activation summary statistics, positive-rate diagnostics (train/test, filtered/unfiltered), and the top activating token snippets (token id, surface form, ±12 token context).
   * `per_feature_analysis_gemma2_summary.csv` mirrors the numeric columns of the JSON (thresholds, activation stats, balanced-train sizes, filtered/unfiltered AUCs, notes) without the heavy text snippets for quicker spreadsheet analysis.

These artifacts make it easy to inspect why a probe achieved a high (or low) AUC by revealing how often the feature fires, what lexical/contextual cues trigger it, and whether the run skipped any stages (e.g., too few positives, no padding run).

Because multiple SAE configurations run sequentially, every artifact listed above is emitted once per config with the corresponding `name` prefix (e.g., `layer25_width16k_l0_116_probe_results_gemma2_filtered.csv`). This keeps results from different layers separate while preserving the same schema.

All outputs saved to CSV for downstream analysis.

---

## 10. Interpretive Goals

The experiment supports several research questions:

* **How lexically identifiable are SAE features?**
* **Do early contextual residual representations encode additional structure relevant to each feature?**
* **Are some SAE features trivially predicted from embeddings, indicating token-identity features?**
* **Do padding tokens or special token mechanics distort probe performance?**
* **How stable are feature interpretations across models of different scale?**

The design enables fair comparisons across features, models, and architectures.
