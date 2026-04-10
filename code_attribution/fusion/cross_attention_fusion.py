import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from code_attribution.sequential import (
    Config as SeqConfig,
    CharVocabulary,
    LexicalFeatureExtractor,
    BiLSTMStyleClassifier,
    AttentionPooling,
    load_data,
    stratified_split,
)
from code_attribution.ast_gnn import (
    ASTConfig,
    ASTGraphBuilder,
    ASTGATClassifier,
)
from code_attribution.fusion.fusion import (
    FusionConfig,
    FusionDataset,
    fusion_collate_fn,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class CrossAttentionConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN = 512
    MAX_AST_NODES = 512

    VOCAB_SIZE = 200
    CHAR_EMBED_DIM = 64
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2

    MAX_NODE_TYPES = 200
    NODE_EMBED_DIM = 64
    GAT_HIDDEN_DIM = 128
    GAT_NUM_HEADS = 4
    GAT_NUM_LAYERS = 3
    GRAPH_EMBED_DIM = 256

    D_MODEL = 256
    NUM_ATTN_HEADS = 8
    ATTN_DROPOUT = 0.1
    FUSION_HIDDEN_DIM = 256
    DROPOUT = 0.3

    BATCH_SIZE = 8
    EPOCHS = 25
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    USE_CROSS_ATTENTION = True
    USE_CONTRASTIVE = False
    USE_SIAMESE = False
    FUSION_TYPE = "cross_attention"
    MODEL_TYPE = "cross_attention_fusion"


class CrossAttentionFusionModel(nn.Module):
    def __init__(
        self,
        seq_model: BiLSTMStyleClassifier,
        gnn_model: ASTGATClassifier,
        seq_feature_dim: int = 512,
        gat_out_dim: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        num_classes: int = 20,
        attn_dropout: float = 0.1,
        dropout: float = 0.3,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()

        self.seq_model = seq_model
        self.gnn_model = gnn_model
        self.d_model = d_model

        self.seq_proj = nn.Linear(seq_feature_dim, d_model)
        self.ast_proj = nn.Linear(gat_out_dim, d_model)

        self.seq_to_ast_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.ast_to_seq_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=attn_dropout, batch_first=True,
        )

        self.seq_norm = nn.LayerNorm(d_model)
        self.ast_norm = nn.LayerNorm(d_model)

        self.seq_attn_pool = AttentionPooling(d_model)

        fused_dim = d_model * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        self._shape_logged = False
        self._init_projections()

    def _init_projections(self) -> None:
        for proj in (self.seq_proj, self.ast_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _interact(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_depth: torch.Tensor,
        graph_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_seq, seq_mask = self.seq_model.get_sequence_features(token_ids, lengths)

        H_ast_nodes, ast_batch = self.gnn_model.get_node_embeddings(
            graph_x, edge_index, graph_depth, graph_batch
        )

        H_ast_dense, ast_mask = to_dense_batch(H_ast_nodes, ast_batch)

        if not self._shape_logged:
            self._shape_logged = True
            print(
                f"[CrossAttnFusion] H_seq={tuple(H_seq.shape)}  "
                f"H_ast_dense={tuple(H_ast_dense.shape)}"
            )

        H_seq = self.seq_proj(H_seq)
        H_ast = self.ast_proj(H_ast_dense)

        seq_kpm = ~seq_mask
        ast_kpm = ~ast_mask

        # Cross-attention (the memory-heavy part)
        H_seq_att, _ = self.seq_to_ast_attn(
            query=H_seq, key=H_ast, value=H_ast,
            key_padding_mask=ast_kpm,
        )

        H_ast_att, _ = self.ast_to_seq_attn(
            query=H_ast, key=H_seq, value=H_seq,
            key_padding_mask=seq_kpm,
        )

        H_seq = self.seq_norm(H_seq + H_seq_att)
        H_ast = self.ast_norm(H_ast + H_ast_att)

        # Free attention intermediates
        del H_seq_att, H_ast_att

        z_seq = self.seq_attn_pool(H_seq, seq_mask)

        ast_float = ast_mask.float().unsqueeze(-1)
        z_ast = (H_ast * ast_float).sum(dim=1) / ast_float.sum(dim=1).clamp(min=1)

        return z_seq, z_ast

    def get_embedding(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_depth: torch.Tensor,
        graph_batch: torch.Tensor,
        graph_valid: Optional[torch.Tensor] = None,
        lex_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_seq, z_ast = self._interact(
            token_ids, lengths, graph_x, edge_index, graph_depth, graph_batch
        )
        return torch.cat([z_seq, z_ast], dim=-1)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_depth: torch.Tensor,
        graph_batch: torch.Tensor,
        graph_valid: Optional[torch.Tensor] = None,
        lex_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_seq, z_ast = self._interact(
            token_ids, lengths, graph_x, edge_index, graph_depth, graph_batch
        )
        z_fused = torch.cat([z_seq, z_ast], dim=-1)
        logits  = self.classifier(z_fused)
        return {"embedding": z_fused, "logits": logits}


class CrossAttentionFusionClassifier(nn.Module):
    def __init__(self, fusion_model: CrossAttentionFusionModel):
        super().__init__()
        self.fusion_model = fusion_model

    def get_embedding(self, *args, **kwargs) -> torch.Tensor:
        return self.fusion_model.get_embedding(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.fusion_model(*args, **kwargs)["logits"]


def _unpack_batch(batch: dict, device: str):
    token_ids   = batch["token_ids"].to(device)
    lengths     = batch["lengths"].to(device)
    labels      = batch["labels"].to(device)
    gb          = batch["graph_batch"].to(device)
    graph_valid = batch["graph_valid"].to(device)
    lex_feats   = batch.get("lex_feats")
    if lex_feats is not None:
        lex_feats = lex_feats.to(device)
    return token_ids, lengths, labels, gb, graph_valid, lex_feats


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    for batch in loader:
        token_ids, lengths, labels, gb, _, _ = _unpack_batch(batch, device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            out    = model(token_ids=token_ids, lengths=lengths,
                           graph_x=gb.x, edge_index=gb.edge_index,
                           graph_depth=gb.depth, graph_batch=gb.batch)
            logits = out["logits"] if isinstance(out, dict) else out
            loss   = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        token_ids, lengths, labels, gb, _, _ = _unpack_batch(batch, device)

        with autocast("cuda", enabled=(device == "cuda")):
            out    = model(token_ids=token_ids, lengths=lengths,
                           graph_x=gb.x, edge_index=gb.edge_index,
                           graph_depth=gb.depth, graph_batch=gb.batch)
            logits = out["logits"] if isinstance(out, dict) else out
            loss   = criterion(logits, labels)

        preds  = logits.float().argmax(1)

        total_loss += loss.float().item() * len(labels)
        correct    += (preds == labels).sum().item()
        total      += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def classification_report(preds, labels, class_names):
    n = len(class_names)
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
    print(sep); print(hdr); print(sep)
    for name, p, r, f, s in rows:
        print(f"{name:<29} {p:>9.3f} {r:>9.3f} {f:>9.3f} {s:>8}")
    print(sep)
    mp = sum(row[1] for row in rows) / n
    mr = sum(row[2] for row in rows) / n
    mf = sum(row[3] for row in rows) / n
    print(f"{'Macro avg':<29} {mp:>9.3f} {mr:>9.3f} {mf:>9.3f} {len(labels):>8}")
    print(sep)


def main():
    config = CrossAttentionConfig()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 60)
    print("  Cross-Attention Fusion (BiLSTM ↔ GAT)")
    print("=" * 60)
    print(f"Device   : {config.DEVICE}")
    print(f"d_model  : {config.D_MODEL}   heads: {config.NUM_ATTN_HEADS}\n")

    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, author2idx = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(f"Split → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}\n")

    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    print(f"Char vocabulary size : {len(vocab)}")

    graph_builder = ASTGraphBuilder(max_nodes=config.MAX_AST_NODES)
    graph_builder.build_vocabulary(
        train_df[config.CODE_COLUMN].tolist(), max_types=config.MAX_NODE_TYPES
    )
    print(f"AST node-type vocab  : {graph_builder.vocab_size}\n")

    def make_ds(split_df, desc):
        print(f"Building {desc} dataset …")
        return FusionDataset(
            codes=split_df[config.CODE_COLUMN].tolist(),
            labels=split_df["label"].tolist(),
            vocab=vocab,
            lex_extractor=None,
            graph_builder=graph_builder,
            max_seq_len=config.MAX_SEQ_LEN,
        )

    collate = partial(fusion_collate_fn, use_lex=False)
    train_loader = DataLoader(make_ds(train_df, "train"),
                              batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader   = DataLoader(make_ds(val_df,   "val"),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=2)
    test_loader  = DataLoader(make_ds(test_df,  "test"),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=2)

    num_classes  = len(top_authors)
    seq_feat_dim = config.LSTM_HIDDEN_DIM * 2
    gat_out_dim  = config.GRAPH_EMBED_DIM // 2

    seq_model = BiLSTMStyleClassifier(
        vocab_size=len(vocab), embed_dim=config.CHAR_EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM, num_classes=num_classes,
        num_layers=config.LSTM_NUM_LAYERS, dropout=config.DROPOUT,
        lex_feature_dim=0,
    )
    gnn_model = ASTGATClassifier(
        num_node_types=graph_builder.vocab_size,
        node_embed_dim=config.NODE_EMBED_DIM, gat_hidden_dim=config.GAT_HIDDEN_DIM,
        num_heads=config.GAT_NUM_HEADS, num_layers=config.GAT_NUM_LAYERS,
        graph_embed_dim=config.GRAPH_EMBED_DIM, num_classes=num_classes,
        dropout=config.DROPOUT,
    )

    model = CrossAttentionFusionModel(
        seq_model=seq_model, gnn_model=gnn_model,
        seq_feature_dim=seq_feat_dim, gat_out_dim=gat_out_dim,
        d_model=config.D_MODEL, num_heads=config.NUM_ATTN_HEADS,
        num_classes=num_classes, attn_dropout=config.ATTN_DROPOUT,
        dropout=config.DROPOUT, fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CrossAttentionFusion parameters: {n_params:,}")

    # AMP scaler for mixed-precision training
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Using mixed-precision (AMP) training to reduce memory\n")

    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}")
    print("─" * 56)

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                      config.DEVICE, scaler=scaler)
        torch.cuda.empty_cache()  # free training intermediates before eval
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, config.DEVICE)
        torch.cuda.empty_cache()
        scheduler.step(vl_acc)

        marker = " <" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}{marker}")

    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_state.items()})

    test_loss, test_acc, preds, labels_list = evaluate(
        model, test_loader, criterion, config.DEVICE
    )
    print(f"Test Loss : {test_loss:.4f}  |  Test Accuracy : {test_acc:.4f}\n")

    idx2author  = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(num_classes)]
    classification_report(preds, labels_list, class_names)

    save_path = "cross_attention_fusion.pt"
    torch.save({
        "model_state":     best_state,
        "vocab":           vocab,
        "node_type_vocab": graph_builder.type2idx,
        "author2idx":      author2idx,
        "config": {k: v for k, v in config.__dict__.items()
                   if not k.startswith("_")},
    }, save_path)
    print(f"\nCheckpoint saved → {save_path}")


if __name__ == "__main__":
    main()
