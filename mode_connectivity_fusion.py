import os
import copy
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

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

warnings.filterwarnings("ignore", category=FutureWarning)


class MCConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN = 2000

    VOCAB_SIZE = 200
    EMBED_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    USE_LEXICAL_FEATURES = True
    DROPOUT = 0.3

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    SEED_A = 42
    SEED_B = 123

    BEZIER_STEPS = 300
    BEZIER_LR = 1e-3

    N_EVAL_POINTS = 11

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lerp_state_dicts(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    result = OrderedDict()
    for key in sd_a:
        result[key] = (
            (1.0 - alpha) * sd_a[key].float() + alpha * sd_b[key].float()
        ).to(sd_a[key].dtype)
    return result


def bezier_state_dicts(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    sd_bend: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    result = OrderedDict()
    c0 = (1 - t) ** 2
    c1 = 2 * t * (1 - t)
    c2 = t**2
    for key in sd_a:
        result[key] = (
            c0 * sd_a[key].float() + c1 * sd_bend[key].float() + c2 * sd_b[key].float()
        ).to(sd_a[key].dtype)
    return result


def sd_to_vector(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in sd.values()])


def vector_to_sd(
    vec: torch.Tensor,
    reference_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = OrderedDict()
    offset = 0
    for key, ref_val in reference_sd.items():
        numel = ref_val.numel()
        result[key] = (
            vec[offset : offset + numel].reshape(ref_val.shape).to(ref_val.dtype)
        )
        offset += numel
    return result


def evaluate_path(
    path_fn: Callable[[float], Dict[str, torch.Tensor]],
    model_factory: Callable[[], nn.Module],
    loader: DataLoader,
    device: str,
    use_lex: bool,
    n_points: int = 11,
    label: str = "",
) -> Tuple[List[float], List[float], List[float]]:
    criterion = nn.CrossEntropyLoss()
    ts, losses, accs = [], [], []

    for i in range(n_points):
        t = i / (n_points - 1)
        sd = path_fn(t)

        model = model_factory()
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})
        model.to(device)

        loss, acc, _, _ = seq_evaluate(model, loader, criterion, device, use_lex)
        ts.append(t)
        losses.append(loss)
        accs.append(acc)

        print(f"  [{label}] t={t:.2f}  loss={loss:.4f}  acc={acc:.4f}")

    return ts, losses, accs


class BezierBendOptimiser:

    def __init__(
        self,
        sd_a: Dict[str, torch.Tensor],
        sd_b: Dict[str, torch.Tensor],
        model_factory: Callable[[], nn.Module],
        device: str,
    ):
        self.sd_a = sd_a
        self.sd_b = sd_b
        self.model_factory = model_factory
        self.device = device

        self.bend_vec = nn.Parameter(
            (sd_to_vector(sd_a) + sd_to_vector(sd_b)) / 2.0
        ).to(device)

        self.vec_a = sd_to_vector(sd_a).to(device).detach()
        self.vec_b = sd_to_vector(sd_b).to(device).detach()
        self.reference_sd = sd_a

    def step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        use_lex: bool,
    ) -> float:
        optimizer.zero_grad()

        t = random.uniform(0.1, 0.9)
        c0 = (1 - t) ** 2
        c1 = 2 * t * (1 - t)
        c2 = t**2

        theta = c0 * self.vec_a + c1 * self.bend_vec + c2 * self.vec_b
        sd = vector_to_sd(theta, self.reference_sd)

        model = self.model_factory()
        model.to(self.device)

        for name, param in model.named_parameters():
            if name in sd:
                param.data.copy_(sd[name].to(self.device))

        token_ids = batch["token_ids"].to(self.device)
        lengths = batch["lengths"].to(self.device)
        labels = batch["labels"].to(self.device)
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(self.device)

        logits = model(token_ids, lengths, lex_feats)
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss_val = loss.item()

        eps = 1e-3
        grad = torch.zeros_like(self.bend_vec)

        n_coords = min(1000, self.bend_vec.numel())
        indices = torch.randperm(self.bend_vec.numel())[:n_coords]

        for idx in indices:
            old_val = self.bend_vec.data[idx].item()

            self.bend_vec.data[idx] = old_val + eps
            theta_plus = c0 * self.vec_a + c1 * self.bend_vec + c2 * self.vec_b
            sd_plus = vector_to_sd(theta_plus, self.reference_sd)
            for name, param in model.named_parameters():
                if name in sd_plus:
                    param.data.copy_(sd_plus[name].to(self.device))
            with torch.no_grad():
                logits_plus = model(token_ids, lengths, lex_feats)
                loss_plus = nn.CrossEntropyLoss()(logits_plus, labels).item()

            self.bend_vec.data[idx] = old_val - eps
            theta_minus = c0 * self.vec_a + c1 * self.bend_vec + c2 * self.vec_b
            sd_minus = vector_to_sd(theta_minus, self.reference_sd)
            for name, param in model.named_parameters():
                if name in sd_minus:
                    param.data.copy_(sd_minus[name].to(self.device))
            with torch.no_grad():
                logits_minus = model(token_ids, lengths, lex_feats)
                loss_minus = nn.CrossEntropyLoss()(logits_minus, labels).item()

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            self.bend_vec.data[idx] = old_val

        with torch.no_grad():
            self.bend_vec -= optimizer.param_groups[0]["lr"] * grad

        return loss_val

    def get_bend_sd(self) -> Dict[str, torch.Tensor]:
        return vector_to_sd(self.bend_vec.detach().cpu(), self.reference_sd)


def optimise_bezier_simple(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    model_factory: Callable[[], nn.Module],
    val_loader: DataLoader,
    device: str,
    use_lex: bool,
    n_candidates: int = 5,
) -> Dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    midpoint = lerp_state_dicts(sd_a, sd_b, 0.5)

    best_loss = float("inf")
    best_bend = midpoint

    for c in range(n_candidates):
        if c == 0:
            bend = midpoint
        else:
            bend = OrderedDict()
            for key in midpoint:
                noise = torch.randn_like(midpoint[key].float()) * 0.01
                bend[key] = (midpoint[key].float() + noise).to(midpoint[key].dtype)

        sd_half = bezier_state_dicts(sd_a, sd_b, bend, t=0.5)
        model = model_factory()
        model.load_state_dict({k: v.to(device) for k, v in sd_half.items()})
        model.to(device)

        loss, acc, _, _ = seq_evaluate(model, val_loader, criterion, device, use_lex)
        print(f"  Candidate {c}: val_loss={loss:.4f}  val_acc={acc:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_bend = bend

    return best_bend


def classification_report(preds, labels, class_names):
    n = len(class_names)
    rows = []
    for i, name in enumerate(class_names):
        tp = sum(p == i and l == i for p, l in zip(preds, labels))
        fp = sum(p == i and l != i for p, l in zip(preds, labels))
        fn = sum(p != i and l == i for p, l in zip(preds, labels))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
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


def main():
    config = MCConfig()

    print("=" * 60)
    print("  Mode Connectivity Fusion")
    print("=" * 60)
    print(f"Device : {config.DEVICE}")
    print(f"Model A seed: {config.SEED_A}  |  Model B seed: {config.SEED_B}\n")

    torch.manual_seed(config.SEED_A)
    random.seed(config.SEED_A)
    np.random.seed(config.SEED_A)

    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, author2idx = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED_A, config.VAL_RATIO, config.TEST_RATIO
    )

    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    lex_ext = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    lex_dim = lex_ext.feature_dim if lex_ext else 0
    use_lex = lex_dim > 0
    num_classes = len(top_authors)

    def make_loader(split_df, shuffle):
        ds = CodeStyleDataset(
            split_df[config.CODE_COLUMN].tolist(),
            split_df["label"].tolist(),
            vocab,
            lex_ext,
            config.MAX_SEQ_LEN,
        )
        return DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=make_collate_fn(use_lex),
            num_workers=2,
        )

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)
    test_loader = make_loader(test_df, False)

    idx2author = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(num_classes)]

    def model_factory():
        return BiLSTMStyleClassifier(
            vocab_size=len(vocab),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            lex_feature_dim=lex_dim,
        )

    criterion = nn.CrossEntropyLoss()

    def train_model(seed: int, label: str) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = model_factory().to(config.DEVICE)
        optimizer = Adam(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

        best_acc, best_sd = 0.0, None
        for epoch in range(1, config.EPOCHS + 1):
            tr_loss, tr_acc = seq_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.DEVICE,
                use_lex,
            )
            vl_loss, vl_acc, _, _ = seq_evaluate(
                model,
                val_loader,
                criterion,
                config.DEVICE,
                use_lex,
            )
            scheduler.step(vl_acc)
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == config.EPOCHS:
                print(
                    f"  [{label}] Epoch {epoch:>3}  "
                    f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}"
                )

        print(f"  [{label}] Best val_acc = {best_acc:.4f}")
        return best_sd

    print("Training Model A …")
    sd_a = train_model(config.SEED_A, "A")

    print("\nTraining Model B …")
    sd_b = train_model(config.SEED_B, "B")

    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in sd_a.items()})
    model.to(config.DEVICE)
    _, acc_a, _, _ = seq_evaluate(model, test_loader, criterion, config.DEVICE, use_lex)

    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in sd_b.items()})
    model.to(config.DEVICE)
    _, acc_b, _, _ = seq_evaluate(model, test_loader, criterion, config.DEVICE, use_lex)

    print(f"\nModel A test_acc = {acc_a:.4f}")
    print(f"Model B test_acc = {acc_b:.4f}")

    print("\n" + "─" * 50)
    print("  Method 1: Linear Interpolation (LERP)")
    print("─" * 50)

    lin_ts, lin_losses, lin_accs = evaluate_path(
        path_fn=lambda t: lerp_state_dicts(sd_a, sd_b, t),
        model_factory=model_factory,
        loader=val_loader,
        device=config.DEVICE,
        use_lex=use_lex,
        n_points=config.N_EVAL_POINTS,
        label="LERP-val",
    )

    best_t_lin = lin_ts[np.argmax(lin_accs)]
    print(f"  Best LERP t={best_t_lin:.2f}  val_acc={max(lin_accs):.4f}")

    best_sd_lin = lerp_state_dicts(sd_a, sd_b, best_t_lin)
    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_sd_lin.items()})
    model.to(config.DEVICE)
    _, test_acc_lin, preds_lin, labels_lin = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex
    )
    print(f"  LERP (t={best_t_lin:.2f}) test_acc = {test_acc_lin:.4f}")

    print("\n" + "─" * 50)
    print("  Method 2: Quadratic Bézier Curve")
    print("─" * 50)

    print("Optimising bend point …")
    bend_sd = optimise_bezier_simple(
        sd_a,
        sd_b,
        model_factory,
        val_loader,
        config.DEVICE,
        use_lex,
        n_candidates=5,
    )

    print("\nEvaluating Bézier path …")
    bez_ts, bez_losses, bez_accs = evaluate_path(
        path_fn=lambda t: bezier_state_dicts(sd_a, sd_b, bend_sd, t),
        model_factory=model_factory,
        loader=val_loader,
        device=config.DEVICE,
        use_lex=use_lex,
        n_points=config.N_EVAL_POINTS,
        label="Bezier-val",
    )

    best_t_bez = bez_ts[np.argmax(bez_accs)]
    print(f"  Best Bézier t={best_t_bez:.2f}  val_acc={max(bez_accs):.4f}")

    best_sd_bez = bezier_state_dicts(sd_a, sd_b, bend_sd, best_t_bez)
    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_sd_bez.items()})
    model.to(config.DEVICE)
    _, test_acc_bez, preds_bez, labels_bez = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex
    )
    print(f"  Bézier (t={best_t_bez:.2f}) test_acc = {test_acc_bez:.4f}")

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  Model A (seed={config.SEED_A}):            test_acc={acc_a:.4f}")
    print(f"  Model B (seed={config.SEED_B}):            test_acc={acc_b:.4f}")
    print(f"  Naive midpoint (α=0.5):           test_acc=", end="")
    mid_sd = lerp_state_dicts(sd_a, sd_b, 0.5)
    model = model_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in mid_sd.items()})
    model.to(config.DEVICE)
    _, mid_acc, _, _ = seq_evaluate(
        model, test_loader, criterion, config.DEVICE, use_lex
    )
    print(f"{mid_acc:.4f}")
    print(f"  Best LERP (t={best_t_lin:.2f}):            test_acc={test_acc_lin:.4f}")
    print(f"  Best Bézier (t={best_t_bez:.2f}):          test_acc={test_acc_bez:.4f}")

    barrier_lin = max(lin_losses) - min(lin_losses[0], lin_losses[-1])
    barrier_bez = max(bez_losses) - min(bez_losses[0], bez_losses[-1])
    print(f"\n  Loss barrier (LERP):   {barrier_lin:.4f}")
    print(f"  Loss barrier (Bézier): {barrier_bez:.4f}")

    if test_acc_bez >= test_acc_lin:
        print(f"\n── Classification Report (Bézier t={best_t_bez:.2f}) ──")
        classification_report(preds_bez, labels_bez, class_names)
    else:
        print(f"\n── Classification Report (LERP t={best_t_lin:.2f}) ──")
        classification_report(preds_lin, labels_lin, class_names)

    save_path = "mode_connectivity_results.pt"
    torch.save(
        {
            "sd_a": sd_a,
            "sd_b": sd_b,
            "bend_sd": bend_sd,
            "best_t_lerp": best_t_lin,
            "best_t_bezier": best_t_bez,
            "lerp_path": {"t": lin_ts, "loss": lin_losses, "acc": lin_accs},
            "bezier_path": {"t": bez_ts, "loss": bez_losses, "acc": bez_accs},
            "author2idx": author2idx,
        },
        save_path,
    )
    print(f"\nMode connectivity results saved → {save_path}")


if __name__ == "__main__":
    main()
