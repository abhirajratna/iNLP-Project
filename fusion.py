import os
import re
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch

from sequential import (
    Config as SeqConfig,
    CharVocabulary,
    LexicalFeatureExtractor,
    BiLSTMStyleClassifier,
    load_data,
    stratified_split,
)
from ast_gnn import (
    ASTConfig,
    ASTGraphBuilder,
    ASTGATClassifier,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class FusionConfig:

    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN = 2000
    MAX_AST_NODES = 2000

    VOCAB_SIZE = 200
    CHAR_EMBED_DIM = 64
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    USE_LEXICAL_FEATURES = True

    MAX_NODE_TYPES = 200
    NODE_EMBED_DIM = 64
    GAT_HIDDEN_DIM = 128
    GAT_NUM_HEADS = 4
    GAT_NUM_LAYERS = 3
    GRAPH_EMBED_DIM = 256

    FUSION_HIDDEN_DIM = 256
    DROPOUT = 0.3

    BATCH_SIZE = 32
    EPOCHS = 25
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FusionDataset:

    def __init__(
        self,
        codes: List[str],
        labels: List[int],
        vocab: CharVocabulary,
        lex_extractor: Optional[LexicalFeatureExtractor],
        graph_builder: ASTGraphBuilder,
        max_seq_len: int,
    ):
        self.codes = codes
        self.labels = labels
        self.vocab = vocab
        self.lex_extractor = lex_extractor
        self.graph_builder = graph_builder
        self.max_seq_len = max_seq_len

        self.graphs: List[Optional[Data]] = []
        skipped = 0
        for code in codes:
            g = graph_builder.code_to_graph(code)
            if g is None:
                skipped += 1
            self.graphs.append(g)
        print(
            f"  FusionDataset: {len(codes)} samples, "
            f"{skipped} AST parse failures (will use dummy graph)"
        )

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int) -> dict:
        code = self.codes[idx]

        token_ids, length = self.vocab.encode(code, self.max_seq_len)

        item = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": length,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.lex_extractor is not None:
            feats = self.lex_extractor.extract(code)
            item["lex_feats"] = torch.tensor(feats, dtype=torch.float)

        graph = self.graphs[idx]
        if graph is not None:
            item["graph"] = graph
            item["graph_valid"] = True
        else:
            item["graph"] = Data(
                x=torch.zeros(1, dtype=torch.long),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                depth=torch.zeros(1, dtype=torch.float),
                num_nodes=1,
            )
            item["graph_valid"] = False

        return item


def fusion_collate_fn(batch: List[dict], use_lex: bool = True) -> dict:
    token_ids = torch.stack([b["token_ids"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long).clamp(min=1)
    labels = torch.stack([b["label"] for b in batch])

    graphs = [b["graph"] for b in batch]
    graph_batch = Batch.from_data_list(graphs)
    graph_valid = torch.tensor([b["graph_valid"] for b in batch], dtype=torch.bool)

    out = {
        "token_ids": token_ids,
        "lengths": lengths,
        "labels": labels,
        "graph_batch": graph_batch,
        "graph_valid": graph_valid,
    }

    if use_lex and "lex_feats" in batch[0]:
        out["lex_feats"] = torch.stack([b["lex_feats"] for b in batch])

    return out


class FeatureFusionClassifier(nn.Module):

    def __init__(
        self,
        seq_vocab_size: int,
        seq_embed_dim: int = 64,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        num_node_types: int = 200,
        node_embed_dim: int = 64,
        gat_hidden_dim: int = 128,
        gat_num_heads: int = 4,
        gat_num_layers: int = 3,
        graph_embed_dim: int = 256,
        fusion_hidden_dim: int = 256,
        num_classes: int = 20,
        dropout: float = 0.3,
        lex_feature_dim: int = 0,
    ):
        super().__init__()

        self.seq_branch = BiLSTMStyleClassifier(
            vocab_size=seq_vocab_size,
            embed_dim=seq_embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_classes=1,
            num_layers=lstm_num_layers,
            dropout=dropout,
            lex_feature_dim=0,
        )
        seq_dim = lstm_hidden_dim * 2

        self.graph_branch = ASTGATClassifier(
            num_node_types=num_node_types,
            node_embed_dim=node_embed_dim,
            gat_hidden_dim=gat_hidden_dim,
            num_heads=gat_num_heads,
            num_layers=gat_num_layers,
            graph_embed_dim=graph_embed_dim,
            num_classes=1,
            dropout=dropout,
        )
        graph_dim = graph_embed_dim

        self.lex_feature_dim = lex_feature_dim
        total_dim = seq_dim + graph_dim + lex_feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def get_seq_embedding(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        _, T = token_ids.shape
        embedded = self.seq_branch.embedding(token_ids)

        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_packed, _ = self.seq_branch.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=T)

        mask = torch.arange(T, device=token_ids.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        context = self.seq_branch.attn_pool(lstm_out, mask)
        return context

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        depth: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.graph_branch.get_embedding(x, edge_index, depth, batch)

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
    ) -> torch.Tensor:

        seq_emb = self.get_seq_embedding(token_ids, lengths)

        graph_emb = self.get_graph_embedding(
            graph_x, edge_index, graph_depth, graph_batch
        )

        if graph_valid is not None:
            graph_emb = graph_emb * graph_valid.float().unsqueeze(-1)

        parts = [seq_emb, graph_emb]
        if lex_feats is not None and self.lex_feature_dim > 0:
            parts.append(lex_feats)

        fused = torch.cat(parts, dim=-1)
        return self.classifier(fused)


def train_epoch(model, loader, optimizer, criterion, device, use_lex):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        gb = batch["graph_batch"].to(device)
        graph_valid = batch["graph_valid"].to(device)
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        optimizer.zero_grad()
        logits = model(
            token_ids=token_ids,
            lengths=lengths,
            graph_x=gb.x,
            edge_index=gb.edge_index,
            graph_depth=gb.depth,
            graph_batch=gb.batch,
            graph_valid=graph_valid,
            lex_feats=lex_feats,
        )
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_lex):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        gb = batch["graph_batch"].to(device)
        graph_valid = batch["graph_valid"].to(device)
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        logits = model(
            token_ids=token_ids,
            lengths=lengths,
            graph_x=gb.x,
            edge_index=gb.edge_index,
            graph_depth=gb.depth,
            graph_batch=gb.batch,
            graph_valid=graph_valid,
            lex_feats=lex_feats,
        )
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(labels)
        correct += (preds == labels).sum().item()
        total += len(labels)
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
    config = FusionConfig()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 60)
    print("  Feature-Level Fusion: Bi-LSTM ‖ GAT")
    print("=" * 60)
    print(f"Device : {config.DEVICE}")
    print(f"Authors: {config.TOP_N_AUTHORS}\n")

    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, author2idx = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(
        f"\nSplit → train: {len(train_df)}  val: {len(val_df)}  "
        f"test: {len(test_df)}\n"
    )

    vocab = CharVocabulary()
    vocab.build(
        train_df[config.CODE_COLUMN].tolist(),
        max_vocab=config.VOCAB_SIZE,
    )
    print(f"Char vocabulary size: {len(vocab)}")

    lex_extractor = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    lex_dim = lex_extractor.feature_dim if lex_extractor else 0
    print(f"Lexical feature dim : {lex_dim}")

    print("Building AST node-type vocabulary …")
    graph_builder = ASTGraphBuilder(max_nodes=config.MAX_AST_NODES)
    graph_builder.build_vocabulary(
        train_df[config.CODE_COLUMN].tolist(),
        max_types=config.MAX_NODE_TYPES,
    )
    print(f"AST node-type vocab : {graph_builder.vocab_size}\n")

    def make_ds(split_df: pd.DataFrame, desc: str) -> FusionDataset:
        print(f"Building {desc} dataset …")
        return FusionDataset(
            codes=split_df[config.CODE_COLUMN].tolist(),
            labels=split_df["label"].tolist(),
            vocab=vocab,
            lex_extractor=lex_extractor,
            graph_builder=graph_builder,
            max_seq_len=config.MAX_SEQ_LEN,
        )

    train_ds = make_ds(train_df, "train")
    val_ds = make_ds(val_df, "val")
    test_ds = make_ds(test_df, "test")

    from functools import partial

    collate = partial(fusion_collate_fn, use_lex=config.USE_LEXICAL_FEATURES)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
    )

    model = FeatureFusionClassifier(
        seq_vocab_size=len(vocab),
        seq_embed_dim=config.CHAR_EMBED_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        num_node_types=graph_builder.vocab_size,
        node_embed_dim=config.NODE_EMBED_DIM,
        gat_hidden_dim=config.GAT_HIDDEN_DIM,
        gat_num_heads=config.GAT_NUM_HEADS,
        gat_num_layers=config.GAT_NUM_LAYERS,
        graph_embed_dim=config.GRAPH_EMBED_DIM,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        num_classes=len(top_authors),
        dropout=config.DROPOUT,
        lex_feature_dim=lex_dim,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFusion model parameters: {n_params:,}")
    print(model)

    optimizer = Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    col_w = 56
    print(
        f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
        f"{'Val Loss':>8}  {'Val Acc':>7}"
    )
    print("─" * col_w)

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.DEVICE,
            config.USE_LEXICAL_FEATURES,
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            config.DEVICE,
            config.USE_LEXICAL_FEATURES,
        )
        scheduler.step(vl_acc)

        marker = " ◀" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
            f"{vl_loss:>8.4f}  {vl_acc:>7.4f}{marker}"
        )

    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print("Restoring best checkpoint …")
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_state.items()})

    test_loss, test_acc, preds, labels_list = evaluate(
        model,
        test_loader,
        criterion,
        config.DEVICE,
        config.USE_LEXICAL_FEATURES,
    )
    print(f"Test Loss : {test_loss:.4f}  |  Test Accuracy : {test_acc:.4f}\n")

    idx2author = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(len(top_authors))]
    classification_report(preds, labels_list, class_names)

    save_path = "feature_fusion_classifier.pt"
    torch.save(
        {
            "model_state": best_state,
            "vocab": vocab,
            "node_type_vocab": graph_builder.type2idx,
            "author2idx": author2idx,
            "lex_feature_dim": lex_dim,
            "config": {
                k: v for k, v in config.__dict__.items() if not k.startswith("_")
            },
        },
        save_path,
    )
    print(f"\nCheckpoint saved → {save_path}")


if __name__ == "__main__":
    main()
