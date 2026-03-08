"""
GNN on Abstract Syntax Trees for Code Authorship Attribution
=============================================================
Goal: Learn the **structural fingerprint** of individual programmers.

By training a Graph Attention Network (GAT) on ASTs parsed from C++ source
code, the model captures how each programmer *structures* their logic:
  - Nesting depth / branching patterns
  - Control-flow preferences (for vs while, if-chains vs switch)
  - Function decomposition style
  - Expression complexity / operator tree shapes

This is complementary to the Bi-LSTM sequential branch which captures
lexical style (naming, keywords, formatting).

Architecture
------------
  Source Code  →  tree-sitter (C++ AST)
             →  Graph (node type features + parent↔child & sibling edges)
             →  3 × GATConv layers  →  Global Pool (mean ‖ max)
             →  FC Classifier  →  Author prediction

Dataset : GCJ (Google Code Jam C++ solutions)
Column  : flines  → raw source code
Label   : username → author to predict
"""

import os
import re
import random
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class ASTConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_PATH               = "datasets/"
    CODE_COLUMN             = "flines"
    AUTHOR_COLUMN           = "username"
    TOP_N_AUTHORS           = 20
    MIN_SAMPLES_PER_AUTHOR  = 5
    MAX_AST_NODES           = 2000    # truncate very large ASTs

    # ── AST node vocabulary ───────────────────────────────────────────────────
    MAX_NODE_TYPES          = 200     # max distinct AST node types to track

    # ── Model ─────────────────────────────────────────────────────────────────
    NODE_EMBED_DIM  = 64              # embedding dim for node-type vocabulary
    GAT_HIDDEN_DIM  = 128             # hidden dim per GAT layer
    GAT_NUM_HEADS   = 4               # attention heads in GAT layers
    GAT_NUM_LAYERS  = 3               # number of GAT layers
    GRAPH_EMBED_DIM = 256             # final graph-level embedding dimension
    DROPOUT         = 0.3

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE   = 32
    EPOCHS       = 20
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED         = 42

    # ── Split ratios ──────────────────────────────────────────────────────────
    VAL_RATIO  = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# AST Parser & Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

class ASTGraphBuilder:
    """
    Parses C++ source code into an Abstract Syntax Tree using tree-sitter,
    then converts the AST into a PyTorch Geometric graph.

    Node features
    -------------
    Each node in the AST becomes a graph node.  Its feature is the integer
    index of its *node type* (e.g. ``function_definition`` → 42), which is
    later passed through an embedding layer inside the GNN.

    Edge construction
    -----------------
    We add three kinds of edges (all bidirectional):
      1. **Parent → Child** (and reverse):  captures hierarchical nesting.
      2. **Next-Sibling**:  connects consecutive children of the same parent
         so the GNN can learn about statement ordering.

    Truncation
    ----------
    Very large ASTs (auto-generated code, huge solutions) are truncated to
    ``max_nodes`` by a BFS traversal, keeping the shallowest / most
    structurally informative nodes.
    """

    def __init__(self, max_nodes: int = 2000):
        self.max_nodes = max_nodes

        # Initialise tree-sitter C++ parser
        self._cpp_language = Language(tscpp.language())
        self._parser = Parser(self._cpp_language)

        # Node-type vocabulary (built from training data)
        self.type2idx: Dict[str, int] = {}
        self.idx2type: Dict[int, str] = {}

    # ── Vocabulary ────────────────────────────────────────────────────────

    def build_vocabulary(self, codes: List[str], max_types: int = 200) -> None:
        """
        Scan training code to collect all AST node types and build a
        mapping.  Index 0 is reserved for <UNK>.
        """
        counter: Counter = Counter()
        for code in codes:
            tree = self._parse(code)
            if tree is None:
                continue
            self._count_types(tree.root_node, counter)

        # Keep the most common types
        most_common = [t for t, _ in counter.most_common(max_types - 1)]
        vocab = ["<UNK>"] + most_common
        self.type2idx = {t: i for i, t in enumerate(vocab)}
        self.idx2type = {i: t for t, i in self.type2idx.items()}

    def _count_types(self, node, counter: Counter) -> None:
        counter[node.type] += 1
        for child in node.children:
            self._count_types(child, counter)

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse(self, code: str):
        """Parse C++ code; return tree or None on failure."""
        try:
            code_bytes = code.encode("utf-8", errors="replace")
            tree = self._parser.parse(code_bytes)
            return tree
        except Exception:
            return None

    # ── Graph Construction ────────────────────────────────────────────────

    def code_to_graph(self, code: str) -> Optional[Data]:
        """
        Convert a C++ source string into a ``torch_geometric.data.Data``
        object ready for batching.

        Returns ``None`` if parsing fails or the AST is trivially small.
        """
        tree = self._parse(code)
        if tree is None:
            return None

        root = tree.root_node
        if root.child_count == 0:
            return None

        # BFS to collect nodes up to max_nodes, tracking depth as we go
        node_list   = []          # tree-sitter Node objects in BFS order
        node_ids: dict = {}       # id(Node) → index in node_list
        depth_list  = []          # depth of each node
        queue       = [(root, 0)] # (node, depth)

        while queue and len(node_list) < self.max_nodes:
            node, depth = queue.pop(0)
            idx = len(node_list)
            node_ids[id(node)] = idx
            node_list.append(node)
            depth_list.append(depth)
            for child in node.children:
                if len(node_list) + len(queue) < self.max_nodes:
                    queue.append((child, depth + 1))

        n_nodes = len(node_list)
        if n_nodes < 2:
            return None

        # Node features: type index
        node_types = []
        for node in node_list:
            type_idx = self.type2idx.get(node.type, 0)  # 0 = <UNK>
            node_types.append(type_idx)

        x = torch.tensor(node_types, dtype=torch.long)  # [N]

        # Build edges
        src_list, dst_list = [], []

        for node in node_list:
            if id(node) not in node_ids:
                continue
            parent_idx = node_ids[id(node)]

            prev_child_idx = None
            for child in node.children:
                if id(child) not in node_ids:
                    continue
                child_idx = node_ids[id(child)]

                # Parent ↔ Child (bidirectional)
                src_list.extend([parent_idx, child_idx])
                dst_list.extend([child_idx, parent_idx])

                # Next-sibling edge (left → right, bidirectional)
                if prev_child_idx is not None:
                    src_list.extend([prev_child_idx, child_idx])
                    dst_list.extend([child_idx, prev_child_idx])

                prev_child_idx = child_idx

        if len(src_list) == 0:
            return None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E]

        # Depth feature (normalised to [0, 1]) – gives the GNN nesting info
        depths = torch.tensor(depth_list, dtype=torch.float)
        max_depth = depths.max().item()
        if max_depth > 0:
            depths = depths / max_depth

        return Data(
            x=x,                        # [N]      node-type indices
            edge_index=edge_index,      # [2, E]   edges
            depth=depths,               # [N]      normalised depth
            num_nodes=n_nodes,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.type2idx)


# ─────────────────────────────────────────────────────────────────────────────
# GNN Model
# ─────────────────────────────────────────────────────────────────────────────

class ASTGATClassifier(nn.Module):
    """
    Graph Attention Network for AST-based code authorship attribution.

    Architecture
    ------------
    1. **Node-type embedding** – learnable lookup table mapping each AST
       node type to a dense vector.
    2. **Depth projection** – a small linear layer maps the scalar depth
       feature to the same dimension, added to the node embedding.
    3. **K × GATConv layers** – each layer applies multi-head attention
       over the graph structure.  Between layers we apply ELU activation,
       dropout, and batch normalisation.
    4. **Global pooling** – concatenation of global mean and global max
       pooling produces a fixed-size graph embedding regardless of AST size.
    5. **Classifier head** – 2-layer MLP with dropout → num_classes logits.

    The graph embedding (before the classifier) is accessible via
    ``get_embedding()`` for downstream fusion with the sequential branch.
    """

    def __init__(
        self,
        num_node_types: int,
        node_embed_dim: int  = 64,
        gat_hidden_dim: int  = 128,
        num_heads:      int  = 4,
        num_layers:     int  = 3,
        graph_embed_dim:int  = 256,
        num_classes:    int  = 20,
        dropout:        float = 0.3,
    ):
        super().__init__()

        self.node_embed_dim = node_embed_dim
        self.dropout = dropout

        # ── Node feature layers ───────────────────────────────────────────
        self.node_embedding = nn.Embedding(num_node_types, node_embed_dim)
        self.depth_proj     = nn.Linear(1, node_embed_dim)

        # ── GAT layers ────────────────────────────────────────────────────
        self.gat_layers = nn.ModuleList()
        self.gat_norms  = nn.ModuleList()

        in_dim = node_embed_dim
        for i in range(num_layers):
            # Last layer uses 1 head and outputs graph_embed_dim // 2
            if i == num_layers - 1:
                out_dim = graph_embed_dim // 2
                heads   = 1
                concat  = False
            else:
                out_dim = gat_hidden_dim
                heads   = num_heads
                concat  = True

            self.gat_layers.append(
                GATConv(
                    in_channels  = in_dim,
                    out_channels = out_dim,
                    heads        = heads,
                    concat       = concat,
                    dropout      = dropout,
                )
            )
            # After concat, output is out_dim * heads (if concat=True)
            actual_out = out_dim * heads if concat else out_dim
            self.gat_norms.append(nn.BatchNorm1d(actual_out))
            in_dim = actual_out

        # After final GAT: in_dim = graph_embed_dim // 2
        # After mean‖max pool: 2 * (graph_embed_dim // 2) = graph_embed_dim
        pool_dim = 2 * (graph_embed_dim // 2)

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, graph_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(graph_embed_dim, num_classes),
        )

    def _node_features(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Combine node-type embedding with depth feature."""
        node_emb  = self.node_embedding(x)               # [N, D]
        depth_emb = self.depth_proj(depth.unsqueeze(-1))  # [N, D]
        return node_emb + depth_emb                       # [N, D]

    def forward(
        self,
        x:          torch.Tensor,   # [N_total]  node-type indices (batched)
        edge_index: torch.Tensor,   # [2, E_total]
        depth:      torch.Tensor,   # [N_total]
        batch:      torch.Tensor,   # [N_total]  graph membership
    ) -> torch.Tensor:
        """Full forward: returns class logits [B, num_classes]."""
        emb = self.get_embedding(x, edge_index, depth, batch)
        return self.classifier(emb)

    def get_embedding(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        depth:      torch.Tensor,
        batch:      torch.Tensor,
    ) -> torch.Tensor:
        """Return the graph-level embedding [B, graph_embed_dim]."""
        h = self._node_features(x, depth)

        for conv, norm in zip(self.gat_layers, self.gat_norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling: mean ‖ max
        h_mean = global_mean_pool(h, batch)   # [B, D]
        h_max  = global_max_pool(h, batch)    # [B, D]
        graph_emb = torch.cat([h_mean, h_max], dim=-1)  # [B, 2D]

        return graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (reuse logic from sequential.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: ASTConfig):
    """Load CSV(s), filter top authors, encode labels."""
    import glob as _glob

    print(f"Loading from: {config.DATA_PATH}")

    if os.path.isdir(config.DATA_PATH):
        csv_files = sorted(_glob.glob(os.path.join(config.DATA_PATH, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {config.DATA_PATH}")
        parts = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(config.DATA_PATH)

    df = df.dropna(subset=[config.CODE_COLUMN, config.AUTHOR_COLUMN])
    df[config.CODE_COLUMN] = df[config.CODE_COLUMN].astype(str)

    counts      = df[config.AUTHOR_COLUMN].value_counts()
    eligible    = counts[counts >= config.MIN_SAMPLES_PER_AUTHOR]
    top_authors = eligible.head(config.TOP_N_AUTHORS).index.tolist()

    df = df[df[config.AUTHOR_COLUMN].isin(top_authors)].copy()
    author2idx  = {a: i for i, a in enumerate(top_authors)}
    df["label"] = df[config.AUTHOR_COLUMN].map(author2idx)

    print(f"Authors  : {len(top_authors)}")
    print(f"Samples  : {len(df)}")
    for a in top_authors:
        print(f"  {a:<30s}  {(df[config.AUTHOR_COLUMN] == a).sum()}")

    return df, top_authors, author2idx


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split (per author)."""
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in df["label"].unique():
        idx = df[df["label"] == label].index.tolist()
        rng.shuffle(idx)
        n       = len(idx)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        test_idx  += idx[:n_test]
        val_idx   += idx[n_test : n_test + n_val]
        train_idx += idx[n_test + n_val :]

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prepare PyG dataset from dataframe
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_list(
    codes: List[str],
    labels: List[int],
    builder: ASTGraphBuilder,
    desc: str = "",
) -> List[Data]:
    """
    Convert a list of code strings into a list of PyG Data objects.
    Skips samples where parsing fails.
    """
    graphs: List[Data] = []
    skipped = 0
    total   = len(codes)

    for i, (code, label) in enumerate(zip(codes, labels)):
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  [{desc}] Parsed {i + 1}/{total}  (skipped {skipped})")

        graph = builder.code_to_graph(code)
        if graph is None:
            skipped += 1
            continue

        graph.y = torch.tensor(label, dtype=torch.long)
        graphs.append(graph)

    print(f"  [{desc}] Done: {len(graphs)} graphs, {skipped} skipped")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.depth, batch.batch)
        loss   = criterion(logits, batch.y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        correct    += (logits.argmax(dim=1) == batch.y).sum().item()
        total      += batch.y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.depth, batch.batch)
        loss   = criterion(logits, batch.y)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * batch.y.size(0)
        correct    += (preds == batch.y).sum().item()
        total      += batch.y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.cpu().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def classification_report(preds, labels, class_names):
    """Per-class precision / recall / F1 report."""
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
    config = ASTConfig()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print(f"Device : {config.DEVICE}")
    print(f"Authors: {config.TOP_N_AUTHORS}  |  Max AST nodes: {config.MAX_AST_NODES}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    df, top_authors, author2idx = load_data(config)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(f"\nSplit  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}\n")

    # ── 2. Build AST graph builder & vocabulary ───────────────────────────
    print("Building AST node-type vocabulary from training data …")
    builder = ASTGraphBuilder(max_nodes=config.MAX_AST_NODES)
    builder.build_vocabulary(
        train_df[config.CODE_COLUMN].tolist(),
        max_types=config.MAX_NODE_TYPES,
    )
    print(f"AST node-type vocabulary size: {builder.vocab_size}\n")

    # ── 3. Convert all splits to graph lists ──────────────────────────────
    print("Parsing ASTs and building graphs …")
    train_graphs = build_graph_list(
        train_df[config.CODE_COLUMN].tolist(),
        train_df["label"].tolist(),
        builder,
        desc="train",
    )
    val_graphs = build_graph_list(
        val_df[config.CODE_COLUMN].tolist(),
        val_df["label"].tolist(),
        builder,
        desc="val",
    )
    test_graphs = build_graph_list(
        test_df[config.CODE_COLUMN].tolist(),
        test_df["label"].tolist(),
        builder,
        desc="test",
    )

    # ── 4. DataLoaders ────────────────────────────────────────────────────
    train_loader = PyGDataLoader(
        train_graphs, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = PyGDataLoader(
        val_graphs, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
    )
    test_loader = PyGDataLoader(
        test_graphs, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
    )

    # ── 5. Model ──────────────────────────────────────────────────────────
    model = ASTGATClassifier(
        num_node_types  = builder.vocab_size,
        node_embed_dim  = config.NODE_EMBED_DIM,
        gat_hidden_dim  = config.GAT_HIDDEN_DIM,
        num_heads       = config.GAT_NUM_HEADS,
        num_layers      = config.GAT_NUM_LAYERS,
        graph_embed_dim = config.GRAPH_EMBED_DIM,
        num_classes     = len(top_authors),
        dropout         = config.DROPOUT,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # ── 6. Optimizer & scheduler ──────────────────────────────────────────
    optimizer = Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # ── 7. Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None

    col_w = 56
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}")
    print("─" * col_w)

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        vl_loss, vl_acc, _, _ = evaluate(
            model, val_loader, criterion, config.DEVICE
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

    # ── 8. Test evaluation ────────────────────────────────────────────────
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print("Restoring best checkpoint …")
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_state.items()})

    test_loss, test_acc, preds, labels_list = evaluate(
        model, test_loader, criterion, config.DEVICE
    )
    print(f"Test Loss : {test_loss:.4f}  |  Test Accuracy : {test_acc:.4f}\n")

    idx2author  = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(len(top_authors))]
    classification_report(preds, labels_list, class_names)

    # ── 9. Save checkpoint ────────────────────────────────────────────────
    save_path = "ast_gat_classifier.pt"
    torch.save(
        {
            "model_state":     best_state,
            "node_type_vocab": builder.type2idx,
            "author2idx":      author2idx,
            "config":          config.__dict__,
        },
        save_path,
    )
    print(f"\nCheckpoint saved → {save_path}")


if __name__ == "__main__":
    main()
