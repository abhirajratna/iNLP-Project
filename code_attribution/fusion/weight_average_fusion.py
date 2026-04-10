import os
import copy
import random
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

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
    train_epoch as seq_train_epoch,
    evaluate as seq_evaluate,
)
from ast_gnn import (
    ASTConfig,
    ASTGraphBuilder,
    ASTGATClassifier,
    build_graph_list,
    train_epoch as gnn_train_epoch,
    evaluate as gnn_evaluate,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class WAConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5

    NUM_SEEDS = 3
    BASE_SEEDS = [42, 123, 7]

    MAX_SEQ_LEN = 2000
    VOCAB_SIZE = 200
    EMBED_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    USE_LEXICAL_FEATURES = True

    MAX_AST_NODES = 2000
    MAX_NODE_TYPES = 200
    NODE_EMBED_DIM = 64
    GAT_HIDDEN_DIM = 128
    GAT_NUM_HEADS = 4
    GAT_NUM_LAYERS = 3
    GRAPH_EMBED_DIM = 256

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def uniform_weight_average(
    state_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    K = len(state_dicts)
    avg = OrderedDict()
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        avg[key] = (stacked.sum(dim=0) / K).to(state_dicts[0][key].dtype)
    return avg


def ema_weight_average(
    state_dicts: List[Dict[str, torch.Tensor]],
    decay: float = 0.9,
) -> Dict[str, torch.Tensor]:
    K = len(state_dicts)
    raw_weights = [decay ** (K - 1 - i) for i in range(K)]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    avg = OrderedDict()
    for key in state_dicts[0]:
        avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts)).to(
            state_dicts[0][key].dtype
        )
    return avg


def greedy_soup(
    state_dicts: List[Dict[str, torch.Tensor]],
    val_accuracies: List[float],
    eval_fn,
    model_factory,
    device: str,
) -> Dict[str, torch.Tensor]:
    order = sorted(
        range(len(state_dicts)), key=lambda i: val_accuracies[i], reverse=True
    )

    soup = [state_dicts[order[0]]]
    best_acc = val_accuracies[order[0]]
    print(f"  Soup starts with checkpoint {order[0]} " f"(val_acc={best_acc:.4f})")

    for idx in order[1:]:
        candidate = soup + [state_dicts[idx]]
        merged = uniform_weight_average(candidate)

        model = model_factory()
        model.load_state_dict({k: v.to(device) for k, v in merged.items()})
        model.to(device)

        acc = eval_fn(model)
        if acc >= best_acc:
            soup.append(state_dicts[idx])
            best_acc = acc
            print(
                f"  + Added checkpoint {idx} → "
                f"soup size {len(soup)}, val_acc={acc:.4f}"
            )
        else:
            print(
                f"  - Rejected checkpoint {idx} "
                f"(val_acc={acc:.4f} < {best_acc:.4f})"
            )

    return uniform_weight_average(soup)


def train_bilstm_multi_seed(config: WAConfig):

    torch.manual_seed(config.BASE_SEEDS[0])
    random.seed(config.BASE_SEEDS[0])
    np.random.seed(config.BASE_SEEDS[0])

    seq_cfg = SeqConfig()
    seq_cfg.DATA_PATH = config.DATA_PATH
    seq_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    seq_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    df, top_authors, author2idx = load_data(seq_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.BASE_SEEDS[0], config.VAL_RATIO, config.TEST_RATIO
    )
    print(
        f"Split → train: {len(train_df)}  val: {len(val_df)}  " f"test: {len(test_df)}"
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

    state_dicts = []
    val_accuracies = []

    for run, seed in enumerate(config.BASE_SEEDS[: config.NUM_SEEDS]):
        print(f"\n── BiLSTM Run {run+1}/{config.NUM_SEEDS} (seed={seed}) ──")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = BiLSTMStyleClassifier(
            vocab_size=len(vocab),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=len(top_authors),
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            lex_feature_dim=lex_dim,
        ).to(config.DEVICE)

        optimizer = Adam(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

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
                    f"  Epoch {epoch:>3}  train_acc={tr_acc:.4f}  "
                    f"val_acc={vl_acc:.4f}"
                )

        state_dicts.append(best_sd)
        val_accuracies.append(best_acc)
        print(f"  Best val_acc = {best_acc:.4f}")

    def eval_seq_model(model):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        _, acc, _, _ = seq_evaluate(
            model, val_loader, criterion, config.DEVICE, use_lex
        )
        return acc

    def seq_model_factory():
        return BiLSTMStyleClassifier(
            vocab_size=len(vocab),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=len(top_authors),
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            lex_feature_dim=lex_dim,
        )

    return (
        state_dicts,
        val_accuracies,
        eval_seq_model,
        seq_model_factory,
        top_authors,
        author2idx,
        test_loader,
        vocab,
        lex_dim,
        use_lex,
    )


def train_gat_multi_seed(config: WAConfig):

    torch.manual_seed(config.BASE_SEEDS[0])
    random.seed(config.BASE_SEEDS[0])
    np.random.seed(config.BASE_SEEDS[0])

    ast_cfg = ASTConfig()
    ast_cfg.DATA_PATH = config.DATA_PATH
    ast_cfg.TOP_N_AUTHORS = config.TOP_N_AUTHORS
    ast_cfg.MIN_SAMPLES_PER_AUTHOR = config.MIN_SAMPLES_PER_AUTHOR

    from ast_gnn import load_data as gnn_load_data

    df, top_authors, author2idx = gnn_load_data(ast_cfg)
    train_df, val_df, test_df = stratified_split(
        df, config.BASE_SEEDS[0], config.VAL_RATIO, config.TEST_RATIO
    )

    builder = ASTGraphBuilder(max_nodes=config.MAX_AST_NODES)
    builder.build_vocabulary(
        train_df[config.CODE_COLUMN].tolist(),
        max_types=config.MAX_NODE_TYPES,
    )

    print("Building graph data …")
    train_graphs = build_graph_list(
        train_df[config.CODE_COLUMN].tolist(),
        train_df["label"].tolist(),
        builder,
        "train",
    )
    val_graphs = build_graph_list(
        val_df[config.CODE_COLUMN].tolist(),
        val_df["label"].tolist(),
        builder,
        "val",
    )
    test_graphs = build_graph_list(
        test_df[config.CODE_COLUMN].tolist(),
        test_df["label"].tolist(),
        builder,
        "test",
    )

    train_loader = PyGDataLoader(
        train_graphs, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = PyGDataLoader(
        val_graphs, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = PyGDataLoader(
        test_graphs, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    state_dicts = []
    val_accuracies = []

    for run, seed in enumerate(config.BASE_SEEDS[: config.NUM_SEEDS]):
        print(f"\n── GAT Run {run+1}/{config.NUM_SEEDS} (seed={seed}) ──")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = ASTGATClassifier(
            num_node_types=builder.vocab_size,
            node_embed_dim=config.NODE_EMBED_DIM,
            gat_hidden_dim=config.GAT_HIDDEN_DIM,
            num_heads=config.GAT_NUM_HEADS,
            num_layers=config.GAT_NUM_LAYERS,
            graph_embed_dim=config.GRAPH_EMBED_DIM,
            num_classes=len(top_authors),
            dropout=config.DROPOUT,
        ).to(config.DEVICE)

        optimizer = Adam(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc, best_sd = 0.0, None
        for epoch in range(1, config.EPOCHS + 1):
            tr_loss, tr_acc = gnn_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.DEVICE,
            )
            vl_loss, vl_acc, _, _ = gnn_evaluate(
                model,
                val_loader,
                criterion,
                config.DEVICE,
            )
            scheduler.step(vl_acc)
            if vl_acc > best_acc:
                best_acc = vl_acc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == config.EPOCHS:
                print(
                    f"  Epoch {epoch:>3}  train_acc={tr_acc:.4f}  "
                    f"val_acc={vl_acc:.4f}"
                )

        state_dicts.append(best_sd)
        val_accuracies.append(best_acc)
        print(f"  Best val_acc = {best_acc:.4f}")

    def eval_gnn_model(model):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        _, acc, _, _ = gnn_evaluate(model, val_loader, criterion, config.DEVICE)
        return acc

    def gnn_model_factory():
        return ASTGATClassifier(
            num_node_types=builder.vocab_size,
            node_embed_dim=config.NODE_EMBED_DIM,
            gat_hidden_dim=config.GAT_HIDDEN_DIM,
            num_heads=config.GAT_NUM_HEADS,
            num_layers=config.GAT_NUM_LAYERS,
            graph_embed_dim=config.GRAPH_EMBED_DIM,
            num_classes=len(top_authors),
            dropout=config.DROPOUT,
        )

    return (
        state_dicts,
        val_accuracies,
        eval_gnn_model,
        gnn_model_factory,
        top_authors,
        author2idx,
        test_loader,
        builder,
    )


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
    config = WAConfig()

    print("=" * 60)
    print("  Weight Average Fusion (Parameter Averaging)")
    print("=" * 60)
    print(f"Device : {config.DEVICE}")
    print(f"Seeds  : {config.BASE_SEEDS[:config.NUM_SEEDS]}")
    print(f"Epochs per seed: {config.EPOCHS}\n")

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Part A: BiLSTM Weight Averaging                       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    (
        seq_sds,
        seq_val_accs,
        seq_eval_fn,
        seq_factory,
        top_authors,
        author2idx,
        seq_test_loader,
        vocab,
        lex_dim,
        use_lex,
    ) = train_bilstm_multi_seed(config)

    idx2author = {v: k for k, v in author2idx.items()}
    class_names = [idx2author[i] for i in range(len(top_authors))]

    criterion = nn.CrossEntropyLoss()

    print("\n── Uniform Weight Average (BiLSTM) ──")
    avg_sd = uniform_weight_average(seq_sds)
    model = seq_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in avg_sd.items()})
    model.to(config.DEVICE)
    _, test_acc, preds, labels_list = seq_evaluate(
        model,
        seq_test_loader,
        criterion,
        config.DEVICE,
        use_lex,
    )
    print(f"  Uniform WA  test_acc = {test_acc:.4f}")

    print("\n── EMA Weight Average (BiLSTM) ──")
    ema_sd = ema_weight_average(seq_sds, decay=0.8)
    model = seq_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in ema_sd.items()})
    model.to(config.DEVICE)
    _, test_acc_ema, _, _ = seq_evaluate(
        model,
        seq_test_loader,
        criterion,
        config.DEVICE,
        use_lex,
    )
    print(f"  EMA WA      test_acc = {test_acc_ema:.4f}")

    print("\n── Greedy Soup (BiLSTM) ──")
    soup_sd = greedy_soup(
        seq_sds,
        seq_val_accs,
        seq_eval_fn,
        seq_factory,
        config.DEVICE,
    )
    model = seq_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in soup_sd.items()})
    model.to(config.DEVICE)
    _, test_acc_soup, preds_soup, labels_soup = seq_evaluate(
        model,
        seq_test_loader,
        criterion,
        config.DEVICE,
        use_lex,
    )
    print(f"  Greedy Soup test_acc = {test_acc_soup:.4f}")

    print("\n── Classification Report (Best BiLSTM WA) ──")
    classification_report(preds_soup, labels_soup, class_names)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Part B: GAT Weight Averaging                          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    (
        gnn_sds,
        gnn_val_accs,
        gnn_eval_fn,
        gnn_factory,
        _,
        _,
        gnn_test_loader,
        builder,
    ) = train_gat_multi_seed(config)

    print("\n── Uniform Weight Average (GAT) ──")
    avg_sd = uniform_weight_average(gnn_sds)
    model = gnn_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in avg_sd.items()})
    model.to(config.DEVICE)
    _, test_acc_g, _, _ = gnn_evaluate(
        model,
        gnn_test_loader,
        criterion,
        config.DEVICE,
    )
    print(f"  Uniform WA  test_acc = {test_acc_g:.4f}")

    print("\n── EMA Weight Average (GAT) ──")
    ema_sd = ema_weight_average(gnn_sds, decay=0.8)
    model = gnn_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in ema_sd.items()})
    model.to(config.DEVICE)
    _, test_acc_g_ema, _, _ = gnn_evaluate(
        model,
        gnn_test_loader,
        criterion,
        config.DEVICE,
    )
    print(f"  EMA WA      test_acc = {test_acc_g_ema:.4f}")

    print("\n── Greedy Soup (GAT) ──")
    soup_sd = greedy_soup(
        gnn_sds,
        gnn_val_accs,
        gnn_eval_fn,
        gnn_factory,
        config.DEVICE,
    )
    model = gnn_factory()
    model.load_state_dict({k: v.to(config.DEVICE) for k, v in soup_sd.items()})
    model.to(config.DEVICE)
    _, test_acc_g_soup, preds_g, labels_g = gnn_evaluate(
        model,
        gnn_test_loader,
        criterion,
        config.DEVICE,
    )
    print(f"  Greedy Soup test_acc = {test_acc_g_soup:.4f}")

    print("\n── Classification Report (Best GAT WA) ──")
    classification_report(preds_g, labels_g, class_names)

    save_path = "weight_average_results.pt"
    torch.save(
        {
            "bilstm_uniform": uniform_weight_average(seq_sds),
            "bilstm_ema": ema_weight_average(seq_sds, decay=0.8),
            "gat_uniform": uniform_weight_average(gnn_sds),
            "gat_ema": ema_weight_average(gnn_sds, decay=0.8),
            "author2idx": author2idx,
        },
        save_path,
    )
    print(f"\nWeight-average checkpoints saved → {save_path}")


if __name__ == "__main__":
    main()
