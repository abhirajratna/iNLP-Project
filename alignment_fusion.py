"""
Alignment-Based Parameter Fusion for Code Authorship Attribution
================================================================
Model-Level Fusion — Sub-type 2b: Alignment before Weight Averaging

Different neural networks trained from different initialisations may learn
the **same features** but store them in **different neurons** (the
"permutation symmetry" of neural networks).  Naively averaging their
weights can therefore be destructive.

Alignment first matches (permutes) the neurons of one network to best
correspond to those of a reference network, then averages the aligned
parameters — preserving the learned representations much better than
plain weight averaging.

Methods
-------
1. **Activation-Based Alignment**
   Run a calibration set through both models, record per-neuron activation
   statistics, then solve a linear assignment problem (Hungarian algorithm)
   to find the best neuron-to-neuron matching between models.

2. **Weight Correlation Alignment**
   Match neurons by the cosine similarity of their incoming weight vectors
   (no data needed, but less accurate for deeper layers).

After alignment, the standard uniform weight average is applied.

Architecture target
-------------------
Demonstrated on the BiLSTM branch (same-architecture K-model setting).
The same technique is applicable to the GAT branch.

Usage
-----
    python alignment_fusion.py
"""

import os
import copy
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment

from sequential import (
    Config as SeqConfig,
    CharVocabulary,
    LexicalFeatureExtractor,
    BiLSTMStyleClassifier,
    CodeStyleDataset,
    load_data,
    stratified_split,
    make_collate_fn,
    train_epoch as seq_train_epoch,
    evaluate as seq_evaluate,
)
from weight_average_fusion import (
    WAConfig,
    uniform_weight_average,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class AlignConfig(WAConfig):
    """Inherits training config, adds alignment-specific settings."""
    # Number of calibration batches for activation alignment
    CALIB_BATCHES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Activation Hook Collector
# ─────────────────────────────────────────────────────────────────────────────

class ActivationCollector:
    """
    Hooks into specified layers to collect per-neuron activation statistics
    (mean activation over a calibration set).
    """

    def __init__(self):
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self._hooks = []

    def register(self, model: nn.Module, layer_names: List[str]):
        """Attach forward hooks to collect activations."""
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)
                self.activations[name] = []

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # output may be a tensor or tuple; take the tensor
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Flatten to [batch, neurons] if needed
            if out.dim() > 2:
                out = out.reshape(out.size(0), -1)
            elif out.dim() == 1:
                out = out.unsqueeze(0)
            self.activations[name].append(out.detach().cpu())
        return hook_fn

    def get_mean_activations(self) -> Dict[str, torch.Tensor]:
        """Return mean activation per neuron [neurons] for each layer."""
        result = {}
        for name, acts in self.activations.items():
            if not acts:
                continue
            cat = torch.cat(acts, dim=0)  # [total_samples, neurons]
            result[name] = cat.mean(dim=0)  # [neurons]
        return result

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.activations = {}
        self.remove_hooks()


# ─────────────────────────────────────────────────────────────────────────────
# Permutation finding
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_matrix_activations(
    acts_ref: torch.Tensor,    # [N] mean activations of reference model
    acts_target: torch.Tensor, # [N] mean activations of target model
) -> np.ndarray:
    """
    Cost matrix based on L2 distance between per-neuron activation profiles.
    For a richer signal, use the full activation vectors [samples, neurons]
    and compute pairwise cosine distances.
    """
    # Reshape to column vectors for broadcasting
    ref = acts_ref.unsqueeze(1).float()     # [N, 1]
    tgt = acts_target.unsqueeze(0).float()  # [1, N]
    cost = (ref - tgt).pow(2).numpy()       # [N, N]
    return cost


def compute_cost_matrix_weights(
    W_ref: torch.Tensor,      # [out_features, in_features]
    W_target: torch.Tensor,   # [out_features, in_features]
) -> np.ndarray:
    """
    Cost matrix based on negative cosine similarity of weight vectors.
    Each row of W is the incoming weight vector for one neuron.
    """
    W_ref_n    = W_ref.float()
    W_target_n = W_target.float()

    # Normalise rows
    W_ref_n    = W_ref_n / (W_ref_n.norm(dim=1, keepdim=True) + 1e-8)
    W_target_n = W_target_n / (W_target_n.norm(dim=1, keepdim=True) + 1e-8)

    # Cosine similarity → cost = 1 - similarity
    sim  = W_ref_n @ W_target_n.T  # [N, N]
    cost = (1.0 - sim).numpy()
    return cost


def find_permutation(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Solve the linear assignment problem (Hungarian algorithm).
    Returns a permutation array P such that target_neuron[i] should be
    mapped to reference_neuron[P[i]].
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind[i] = which target neuron maps to reference neuron i
    # We need: for each reference position i, which target neuron goes there
    perm = np.zeros(len(row_ind), dtype=np.int64)
    perm[row_ind] = col_ind
    return perm


# ─────────────────────────────────────────────────────────────────────────────
# Apply Permutation to BiLSTM State Dict
# ─────────────────────────────────────────────────────────────────────────────

def permute_bilstm_classifier_layer(
    state_dict: Dict[str, torch.Tensor],
    perm: np.ndarray,
    layer_prefix: str = "classifier.1",
) -> Dict[str, torch.Tensor]:
    """
    Permute the neurons of a Linear layer (and adjust the next layer's inputs).

    For the BiLSTM classifier:
      classifier.1 = Linear(fusion_dim, hidden_dim)   ← permute outputs
      classifier.4 = Linear(hidden_dim, num_classes)   ← permute inputs

    Parameters
    ----------
    state_dict : model state dict (will be modified in-place and returned).
    perm       : permutation array [hidden_dim].
    layer_prefix : e.g. "classifier.1" for the first Linear.
    """
    sd = OrderedDict(state_dict)
    perm_t = torch.from_numpy(perm).long()

    # Permute output neurons of the first layer
    w_key = f"{layer_prefix}.weight"
    b_key = f"{layer_prefix}.bias"
    if w_key in sd:
        sd[w_key] = sd[w_key][perm_t]
    if b_key in sd:
        sd[b_key] = sd[b_key][perm_t]

    # Find the next Linear layer and permute its input dimension
    # classifier structure: Dropout(0), Linear(1), ReLU(2), Dropout(3), Linear(4)
    next_w_key = "classifier.4.weight"
    if next_w_key in sd:
        sd[next_w_key] = sd[next_w_key][:, perm_t]

    return sd


# ─────────────────────────────────────────────────────────────────────────────
# Full Alignment Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def align_bilstm_activation(
    ref_model:   BiLSTMStyleClassifier,
    target_sd:   Dict[str, torch.Tensor],
    model_factory,
    calib_loader: DataLoader,
    device:      str,
    use_lex:     bool,
    num_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Align target model neurons to reference model using activation matching,
    then return the aligned state dict.
    """
    # Build target model
    target_model = model_factory()
    target_model.load_state_dict(
        {k: v.to(device) for k, v in target_sd.items()}
    )
    target_model.to(device).eval()

    # Collect activations for the first hidden layer of the classifier
    layer_name = "classifier.1"  # Linear(fusion_dim → hidden_dim)

    ref_collector = ActivationCollector()
    ref_collector.register(ref_model.eval(), [layer_name])

    tgt_collector = ActivationCollector()
    tgt_collector.register(target_model, [layer_name])

    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= num_batches:
                break
            token_ids = batch["token_ids"].to(device)
            lengths   = batch["lengths"].to(device)
            lex_feats = batch.get("lex_feats")
            if lex_feats is not None:
                lex_feats = lex_feats.to(device)
            ref_model(token_ids, lengths, lex_feats)
            target_model(token_ids, lengths, lex_feats)

    ref_acts = ref_collector.get_mean_activations()
    tgt_acts = tgt_collector.get_mean_activations()
    ref_collector.clear()
    tgt_collector.clear()

    if layer_name not in ref_acts or layer_name not in tgt_acts:
        print("  WARNING: Could not collect activations; "
              "returning unaligned state dict.")
        return target_sd

    # Compute cost and find permutation
    cost = compute_cost_matrix_activations(
        ref_acts[layer_name], tgt_acts[layer_name]
    )
    perm = find_permutation(cost)

    # Apply permutation
    aligned_sd = permute_bilstm_classifier_layer(
        target_sd, perm, layer_prefix=layer_name
    )

    return aligned_sd


def align_bilstm_weight_correlation(
    ref_sd:   Dict[str, torch.Tensor],
    target_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Align target model to reference using weight-correlation matching.
    No calibration data needed.
    """
    layer_key = "classifier.1.weight"

    if layer_key not in ref_sd or layer_key not in target_sd:
        print("  WARNING: Layer key not found; returning unaligned state dict.")
        return target_sd

    cost = compute_cost_matrix_weights(ref_sd[layer_key], target_sd[layer_key])
    perm = find_permutation(cost)

    aligned_sd = permute_bilstm_classifier_layer(
        target_sd, perm, layer_prefix="classifier.1"
    )
    return aligned_sd


# ─────────────────────────────────────────────────────────────────────────────
# Classification Report
# ─────────────────────────────────────────────────────────────────────────────

def classification_report(preds, labels, class_names):
    n    = len(class_names)
    rows = []
    for i, name in enumerate(class_names):
        tp = sum(p == i and l == i for p, l in zip(preds, labels))
        fp = sum(p == i and l != i for p, l in zip(preds, labels))
        fn = sum(p != i and l == i for p, l in zip(preds, labels))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((name[:28], prec, rec, f1, sum(l == i for l in labels)))

    hdr = f"{'Author':<29} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}"
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for name, p, r, f, s in rows:
        print(f"{name:<29} {p:>9.3f} {r:>9.3f} {f:>9.3f} {s:>8}")
    print(sep)
    mp = sum(r[1] for r in rows) / n
    mr = sum(r[2] for r in rows) / n
    mf = sum(r[3] for r in rows) / n
    print(f"{'Macro avg':<29} {mp:>9.3f} {mr:>9.3f} {mf:>9.3f} {len(labels):>8}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = AlignConfig()

    torch.manual_seed(config.BASE_SEEDS[0])
    random.seed(config.BASE_SEEDS[0])
    np.random.seed(config.BASE_SEEDS[0])

    print("=" * 60)
    print("  Alignment-Based Parameter Fusion")
    print("=" * 60)
    print(f"Device : {config.DEVICE}")
    print(f"Seeds  : {config.BASE_SEEDS[:config.NUM_SEEDS]}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, author2idx = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.BASE_SEEDS[0], config.VAL_RATIO, config.TEST_RATIO
    )

    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    lex_ext = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    lex_dim = lex_ext.feature_dim if lex_ext else 0
    use_lex = lex_dim > 0

    def make_loader(split_df, shuffle):
        ds = CodeStyleDataset(
            split_df[config.CODE_COLUMN].tolist(),
            split_df["label"].tolist(),
            vocab, lex_ext, config.MAX_SEQ_LEN,
        )
        return DataLoader(
            ds, batch_size=config.BATCH_SIZE,
            shuffle=shuffle, collate_fn=make_collate_fn(use_lex),
            num_workers=2,
        )

    train_loader = make_loader(train_df, True)
    val_loader   = make_loader(val_df, False)
    test_loader  = make_loader(test_df, False)

    num_classes = len(top_authors)
    idx2author  = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(num_classes)]

    def model_factory():
        return BiLSTMStyleClassifier(
            vocab_size      = len(vocab),
            embed_dim       = config.EMBED_DIM,
            hidden_dim      = config.HIDDEN_DIM,
            num_classes     = num_classes,
            num_layers      = config.NUM_LAYERS,
            dropout         = config.DROPOUT,
            lex_feature_dim = lex_dim,
        )

    # ── 2. Train K BiLSTM models ─────────────────────────────────────────
    state_dicts    = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()

    for run, seed in enumerate(config.BASE_SEEDS[:config.NUM_SEEDS]):
        print(f"\n── Training BiLSTM (seed={seed}) ──")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = model_factory().to(config.DEVICE)
        optimizer = Adam(model.parameters(), lr=config.LR,
                         weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=3, factor=0.5)

        best_acc, best_sd = 0.0, None
        for epoch in range(1, config.EPOCHS + 1):
            tr_loss, tr_acc = seq_train_epoch(
                model, train_loader, optimizer, criterion,
                config.DEVICE, use_lex,
            )
            vl_loss, vl_acc, _, _ = seq_evaluate(
                model, val_loader, criterion, config.DEVICE, use_lex,
            )
            scheduler.step(vl_acc)
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_sd = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == config.EPOCHS:
                print(f"  Epoch {epoch:>3}  train_acc={tr_acc:.4f}  "
                      f"val_acc={vl_acc:.4f}")

        state_dicts.append(best_sd)
        val_accuracies.append(best_acc)
        print(f"  Best val_acc = {best_acc:.4f}")

    # ── 3. Baseline: naive weight average ─────────────────────────────────
    print("\n── Baseline: Naive Weight Average ──")
    naive_sd = uniform_weight_average(state_dicts)
    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in naive_sd.items()})
    model.to(config.DEVICE)
    _, naive_acc, _, _ = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex,
    )
    print(f"  Naive WA  test_acc = {naive_acc:.4f}")

    # ── 4. Activation-based alignment + WA ────────────────────────────────
    print("\n── Activation-Based Alignment + WA ──")
    # Use the first model as reference
    ref_model = model_factory()
    ref_model.load_state_dict(
        {k: v.to(config.DEVICE) for k, v in state_dicts[0].items()}
    )
    ref_model.to(config.DEVICE).eval()

    aligned_sds = [state_dicts[0]]  # reference stays as-is
    for i in range(1, len(state_dicts)):
        print(f"  Aligning model {i} to reference (activation matching) …")
        aligned = align_bilstm_activation(
            ref_model, state_dicts[i], model_factory,
            val_loader, config.DEVICE, use_lex,
            num_batches=config.CALIB_BATCHES,
        )
        aligned_sds.append(aligned)

    act_avg_sd = uniform_weight_average(aligned_sds)
    model = model_factory()
    model.load_state_dict(
        {k: v.to(config.DEVICE) for k, v in act_avg_sd.items()}
    )
    model.to(config.DEVICE)
    _, act_acc, preds_act, labels_act = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex,
    )
    print(f"  Activation-Aligned WA  test_acc = {act_acc:.4f}")

    # ── 5. Weight-correlation alignment + WA ──────────────────────────────
    print("\n── Weight-Correlation Alignment + WA ──")
    wt_aligned_sds = [state_dicts[0]]
    for i in range(1, len(state_dicts)):
        print(f"  Aligning model {i} to reference (weight correlation) …")
        aligned = align_bilstm_weight_correlation(
            state_dicts[0], state_dicts[i]
        )
        wt_aligned_sds.append(aligned)

    wt_avg_sd = uniform_weight_average(wt_aligned_sds)
    model = model_factory()
    model.load_state_dict(
        {k: v.to(config.DEVICE) for k, v in wt_avg_sd.items()}
    )
    model.to(config.DEVICE)
    _, wt_acc, preds_wt, labels_wt = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex,
    )
    print(f"  Weight-Corr Aligned WA test_acc = {wt_acc:.4f}")

    # ── 6. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    for i, (sd, vacc) in enumerate(zip(state_dicts, val_accuracies)):
        model = model_factory()
        model.load_state_dict(
            {k: v.to(config.DEVICE) for k, v in sd.items()}
        )
        model.to(config.DEVICE)
        _, tacc, _, _ = seq_evaluate(
            model, test_loader, criterion, config.DEVICE, use_lex
        )
        print(f"  Individual model {i} (seed={config.BASE_SEEDS[i]}): "
              f"test_acc={tacc:.4f}")
    print(f"  Naive Weight Average:               test_acc={naive_acc:.4f}")
    print(f"  Activation-Aligned WA:              test_acc={act_acc:.4f}")
    print(f"  Weight-Correlation Aligned WA:      test_acc={wt_acc:.4f}")

    # ── 7. Detailed report for best ───────────────────────────────────────
    best_method = max(
        [("Activation-Aligned", act_acc, preds_act, labels_act),
         ("Weight-Corr Aligned", wt_acc, preds_wt, labels_wt)],
        key=lambda x: x[1]
    )
    print(f"\n── Classification Report ({best_method[0]}) ──")
    classification_report(best_method[2], best_method[3], class_names)

    # ── 8. Save ───────────────────────────────────────────────────────────
    save_path = "alignment_fusion_results.pt"
    torch.save({
        "activation_aligned_wa": act_avg_sd,
        "weight_corr_aligned_wa": wt_avg_sd,
        "naive_wa": naive_sd,
        "author2idx": author2idx,
    }, save_path)
    print(f"\nAlignment fusion results saved → {save_path}")


if __name__ == "__main__":
    main()
