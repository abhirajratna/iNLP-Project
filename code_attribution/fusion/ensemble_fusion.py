import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from sequential import (
    Config as SeqConfig,
    CharVocabulary,
    LexicalFeatureExtractor,
    BiLSTMStyleClassifier,
    CodeStyleDataset,
    load_data,
    stratified_split,
    make_collate_fn,
)
from ast_gnn import (
    ASTConfig,
    ASTGraphBuilder,
    ASTGATClassifier,
    build_graph_list,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class EnsembleConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN = 2000
    MAX_AST_NODES = 2000

    SEQ_CHECKPOINT = "bilstm_style_classifier.pt"
    GNN_CHECKPOINT = "ast_gat_classifier.pt"

    STACK_HIDDEN_DIM = 64
    STACK_EPOCHS = 50
    STACK_LR = 1e-3

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    BATCH_SIZE = 32


def load_bilstm(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    vocab = ckpt["vocab"]
    author2idx = ckpt["author2idx"]
    lex_feat_dim = ckpt.get("lex_feature_dim", 0)

    model = BiLSTMStyleClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg.get("EMBED_DIM", 64),
        hidden_dim=cfg.get("HIDDEN_DIM", 256),
        num_classes=len(author2idx),
        num_layers=cfg.get("NUM_LAYERS", 2),
        dropout=cfg.get("DROPOUT", 0.3),
        lex_feature_dim=lex_feat_dim,
    )
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    model.to(device).eval()
    return model, vocab, author2idx, lex_feat_dim


def load_gat(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    type2idx = ckpt["node_type_vocab"]
    author2idx = ckpt["author2idx"]

    model = ASTGATClassifier(
        num_node_types=len(type2idx),
        node_embed_dim=cfg.get("NODE_EMBED_DIM", 64),
        gat_hidden_dim=cfg.get("GAT_HIDDEN_DIM", 128),
        num_heads=cfg.get("GAT_NUM_HEADS", 4),
        num_layers=cfg.get("GAT_NUM_LAYERS", 3),
        graph_embed_dim=cfg.get("GRAPH_EMBED_DIM", 256),
        num_classes=len(author2idx),
        dropout=cfg.get("DROPOUT", 0.3),
    )
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    model.to(device).eval()

    builder = ASTGraphBuilder(max_nodes=cfg.get("MAX_AST_NODES", 2000))
    builder.type2idx = type2idx
    builder.idx2type = {v: k for k, v in type2idx.items()}

    return model, builder, author2idx


@torch.no_grad()
def extract_seq_probs(
    model, loader, device, use_lex: bool
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"]
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        logits = model(token_ids, lengths, lex_feats)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def extract_gnn_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.depth, batch.batch)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def simple_average(probs_seq: np.ndarray, probs_gnn: np.ndarray) -> np.ndarray:
    return (probs_seq + probs_gnn) / 2.0


def weighted_average(
    probs_seq: np.ndarray,
    probs_gnn: np.ndarray,
    w_seq: float = 0.5,
) -> np.ndarray:
    return w_seq * probs_seq + (1.0 - w_seq) * probs_gnn


def majority_voting(probs_seq: np.ndarray, probs_gnn: np.ndarray) -> np.ndarray:
    preds_seq = probs_seq.argmax(axis=1)
    preds_gnn = probs_gnn.argmax(axis=1)

    n, c = probs_seq.shape
    votes = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        votes[i, preds_seq[i]] += 1
        votes[i, preds_gnn[i]] += 1

    max_conf_seq = probs_seq.max(axis=1, keepdims=True)
    max_conf_gnn = probs_gnn.max(axis=1, keepdims=True)
    tiebreak = np.where(max_conf_seq > max_conf_gnn, probs_seq, probs_gnn)
    votes = votes + tiebreak * 1e-6

    return votes


def grid_search_weights(
    probs_seq_val: np.ndarray,
    probs_gnn_val: np.ndarray,
    labels_val: np.ndarray,
    steps: int = 21,
) -> float:
    best_w, best_acc = 0.5, 0.0
    for w in np.linspace(0.0, 1.0, steps):
        fused = w * probs_seq_val + (1.0 - w) * probs_gnn_val
        preds = fused.argmax(axis=1)
        acc = (preds == labels_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w


class StackingMLP(nn.Module):

    def __init__(self, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_stacking(
    probs_seq_train: np.ndarray,
    probs_gnn_train: np.ndarray,
    labels_train: np.ndarray,
    probs_seq_val: np.ndarray,
    probs_gnn_val: np.ndarray,
    labels_val: np.ndarray,
    num_classes: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> StackingMLP:
    X_train = torch.tensor(
        np.concatenate([probs_seq_train, probs_gnn_train], axis=1),
        dtype=torch.float,
    ).to(device)
    y_train = torch.tensor(labels_train, dtype=torch.long).to(device)

    X_val = torch.tensor(
        np.concatenate([probs_seq_val, probs_gnn_val], axis=1),
        dtype=torch.float,
    ).to(device)
    y_val = torch.tensor(labels_val, dtype=torch.long).to(device)

    meta = StackingMLP(num_classes, hidden_dim).to(device)
    optimizer = Adam(meta.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        meta.train()
        optimizer.zero_grad()
        logits = meta(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        meta.eval()
        with torch.no_grad():
            val_logits = meta(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in meta.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  [Stacking] Epoch {epoch:>3}  loss={loss.item():.4f}  "
                f"val_acc={val_acc:.4f}"
            )

    print(f"  [Stacking] Best val acc: {best_val_acc:.4f}")
    meta.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    meta.eval()
    return meta


def eval_ensemble(preds: np.ndarray, labels: np.ndarray, name: str) -> float:
    acc = (preds == labels).mean()
    print(f"  {name:<30s}  accuracy = {acc:.4f}")
    return acc


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
    config = EnsembleConfig()
    device = config.DEVICE

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 60)
    print("  Ensemble / Late Fusion (Decision-Level)")
    print("=" * 60)
    print(f"Device: {device}\n")

    if not os.path.isfile(config.SEQ_CHECKPOINT):
        print(f"ERROR: Sequential checkpoint not found: {config.SEQ_CHECKPOINT}")
        print("       Run sequential.py first.")
        return
    if not os.path.isfile(config.GNN_CHECKPOINT):
        print(f"ERROR: GNN checkpoint not found: {config.GNN_CHECKPOINT}")
        print("       Run ast_gnn.py first.")
        return

    print("Loading pre-trained BiLSTM …")
    seq_model, vocab, author2idx, lex_dim = load_bilstm(config.SEQ_CHECKPOINT, device)
    use_lex = lex_dim > 0

    print("Loading pre-trained AST-GAT …")
    gnn_model, graph_builder, _ = load_gat(config.GNN_CHECKPOINT, device)

    num_classes = len(author2idx)
    idx2author = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(num_classes)]

    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, _ = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(
        f"Split → train: {len(train_df)}  val: {len(val_df)}  "
        f"test: {len(test_df)}\n"
    )

    lex_extractor = LexicalFeatureExtractor() if use_lex else None

    def make_seq_loader(split_df):
        ds = CodeStyleDataset(
            codes=split_df[config.CODE_COLUMN].tolist(),
            labels=split_df["label"].tolist(),
            vocab=vocab,
            lex_extractor=lex_extractor,
            max_seq_len=config.MAX_SEQ_LEN,
        )
        collate = make_collate_fn(use_lex)
        return DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=2,
        )

    val_seq_loader = make_seq_loader(val_df)
    test_seq_loader = make_seq_loader(test_df)
    train_seq_loader = make_seq_loader(train_df)

    def make_gnn_loader(split_df, desc):
        graphs = build_graph_list(
            split_df[config.CODE_COLUMN].tolist(),
            split_df["label"].tolist(),
            graph_builder,
            desc=desc,
        )
        return PyGDataLoader(
            graphs,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )

    print("Building graph data …")
    val_gnn_loader = make_gnn_loader(val_df, "val")
    test_gnn_loader = make_gnn_loader(test_df, "test")
    train_gnn_loader = make_gnn_loader(train_df, "train")

    print("\nExtracting predictions from BiLSTM …")
    probs_seq_val, labels_seq_val = extract_seq_probs(
        seq_model, val_seq_loader, device, use_lex
    )
    probs_seq_test, labels_seq_test = extract_seq_probs(
        seq_model, test_seq_loader, device, use_lex
    )
    probs_seq_train, labels_seq_train = extract_seq_probs(
        seq_model, train_seq_loader, device, use_lex
    )

    print("Extracting predictions from AST-GAT …")
    probs_gnn_val, labels_gnn_val = extract_gnn_probs(gnn_model, val_gnn_loader, device)
    probs_gnn_test, labels_gnn_test = extract_gnn_probs(
        gnn_model, test_gnn_loader, device
    )
    probs_gnn_train, labels_gnn_train = extract_gnn_probs(
        gnn_model, train_gnn_loader, device
    )

    print(f"\nSequential (val/test): {len(labels_seq_val)} / {len(labels_seq_test)}")
    print(f"GNN        (val/test): {len(labels_gnn_val)} / {len(labels_gnn_test)}")

    print("\n── Individual Model Baselines ──")
    eval_ensemble(probs_seq_test.argmax(axis=1), labels_seq_test, "BiLSTM (sequential)")
    eval_ensemble(probs_gnn_test.argmax(axis=1), labels_gnn_test, "AST-GAT (graph)")

    n_test = min(len(labels_gnn_test), len(labels_seq_test))
    p_seq = probs_seq_test[:n_test]
    p_gnn = probs_gnn_test[:n_test]
    y_test = labels_seq_test[:n_test]

    n_val = min(len(labels_gnn_val), len(labels_seq_val))
    p_seq_v = probs_seq_val[:n_val]
    p_gnn_v = probs_gnn_val[:n_val]
    y_val = labels_seq_val[:n_val]

    print(f"\nEnsemble evaluation on {n_test} matched test samples:")
    print("─" * 50)

    fused = simple_average(p_seq, p_gnn)
    eval_ensemble(fused.argmax(axis=1), y_test, "Simple Average")

    best_w = grid_search_weights(p_seq_v, p_gnn_v, y_val, steps=21)
    print(f"  (Optimal weight for BiLSTM: {best_w:.2f})")
    fused = weighted_average(p_seq, p_gnn, w_seq=best_w)
    eval_ensemble(fused.argmax(axis=1), y_test, "Weighted Average")

    fused = majority_voting(p_seq, p_gnn)
    eval_ensemble(fused.argmax(axis=1), y_test, "Majority Voting")

    print("\n── Training Stacking Meta-Learner ──")
    n_train = min(len(labels_gnn_train), len(labels_seq_train))
    p_seq_tr = probs_seq_train[:n_train]
    p_gnn_tr = probs_gnn_train[:n_train]
    y_train = labels_seq_train[:n_train]

    meta = train_stacking(
        p_seq_tr,
        p_gnn_tr,
        y_train,
        p_seq_v,
        p_gnn_v,
        y_val,
        num_classes=num_classes,
        hidden_dim=config.STACK_HIDDEN_DIM,
        epochs=config.STACK_EPOCHS,
        lr=config.STACK_LR,
        device=device,
    )

    X_test = torch.tensor(np.concatenate([p_seq, p_gnn], axis=1), dtype=torch.float).to(
        device
    )
    with torch.no_grad():
        stack_preds = meta(X_test).argmax(dim=1).cpu().numpy()
    eval_ensemble(stack_preds, y_test, "Stacking (MLP)")

    print("\n── Classification Report (Weighted Average) ──")
    fused = weighted_average(p_seq, p_gnn, w_seq=best_w)
    best_preds = fused.argmax(axis=1)
    classification_report(best_preds.tolist(), y_test.tolist(), class_names)

    save_path = "ensemble_fusion_results.pt"
    torch.save(
        {
            "optimal_weight_seq": best_w,
            "stacking_state": {k: v.cpu() for k, v in meta.state_dict().items()},
            "num_classes": num_classes,
            "stack_hidden_dim": config.STACK_HIDDEN_DIM,
            "author2idx": author2idx,
        },
        save_path,
    )
    print(f"\nEnsemble results saved → {save_path}")


if __name__ == "__main__":
    main()
