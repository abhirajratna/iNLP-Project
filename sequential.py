import os
import re
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Config:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN = 2000

    USE_LEXICAL_FEATURES = True

    VOCAB_SIZE = 200
    EMBED_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LexicalFeatureExtractor:

    KEYWORDS = [
        "for",
        "while",
        "if",
        "else",
        "switch",
        "return",
        "int",
        "long",
        "string",
        "vector",
        "auto",
        "void",
        "class",
        "struct",
        "typedef",
        "using",
        "namespace",
        "include",
        "define",
        "cout",
        "cin",
        "printf",
        "scanf",
        "break",
        "continue",
    ]

    def __call__(self, code: str) -> List[float]:
        return self.extract(code)

    def extract(self, code: str) -> List[float]:
        lines = code.split("\n")
        tokens = re.findall(r"\b\w+\b", code)
        ids = [t for t in tokens if not t.isdigit() and t not in self.KEYWORDS]

        features: List[float] = []

        n_id = max(len(ids), 1)
        camel = sum(1 for t in ids if re.search(r"[a-z][A-Z]", t))
        snake = sum(1 for t in ids if "_" in t and t != "_")
        all_caps = sum(1 for t in ids if t.isupper() and len(t) > 1)
        features += [camel / n_id, snake / n_id, all_caps / n_id]

        tok_counts = Counter(tokens)
        total_toks = max(len(tokens), 1)
        for kw in self.KEYWORDS:
            features.append(tok_counts.get(kw, 0) / total_toks)

        n_for = tok_counts.get("for", 0)
        n_while = tok_counts.get("while", 0)
        features.append(n_for / max(n_for + n_while, 1))

        indented = [l for l in lines if l and l[0] in (" ", "\t")]
        n_ind = max(len(indented), 1)
        tab_rate = sum(1 for l in indented if l[0] == "\t") / n_ind
        space_rate = sum(1 for l in indented if l[0] == " ") / n_ind
        depths = [len(l) - len(l.lstrip(" ")) for l in lines if l.startswith(" ")]
        avg_indent = (np.mean(depths) / 8.0) if depths else 0.0
        features += [tab_rate, space_rate, min(avg_indent, 1.0)]

        n_lines = max(len(lines), 1)
        features += [
            sum(1 for l in lines if "//" in l) / n_lines,
            code.count("/*") / n_lines,
        ]

        lengths = [len(l) for l in lines if l.strip()]
        if lengths:
            features += [
                min(np.mean(lengths) / 80.0, 1.0),
                min(np.std(lengths) / 40.0, 1.0),
                min(max(lengths) / 120.0, 1.0),
            ]
        else:
            features += [0.0, 0.0, 0.0]

        n_chars = max(len(code), 1)
        features += [
            code.count("{") / n_chars * 100,
            code.count(";") / n_chars * 100,
        ]

        features.append(sum(1 for l in lines if not l.strip()) / n_lines)

        features.append(sum(1 for t in ids if len(t) <= 2) / n_id)

        return features

    @property
    def feature_dim(self) -> int:
        return 3 + len(self.KEYWORDS) + 1 + 3 + 2 + 3 + 2 + 1 + 1


class CharVocabulary:

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

    def build(self, texts: List[str], max_vocab: int = 200) -> None:
        counter = Counter(ch for text in texts for ch in text)
        most_common = [ch for ch, _ in counter.most_common(max_vocab - 2)]
        vocab = [self.PAD, self.UNK] + most_common
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        ids = [self.char2idx.get(ch, 1) for ch in text[:max_len]]
        length = len(ids)
        ids += [0] * (max_len - length)
        return ids, length

    def __len__(self) -> int:
        return len(self.char2idx)


class CodeStyleDataset(Dataset):

    def __init__(
        self,
        codes: List[str],
        labels: List[int],
        vocab: CharVocabulary,
        lex_extractor: Optional[LexicalFeatureExtractor],
        max_seq_len: int,
    ):
        self.codes = codes
        self.labels = labels
        self.vocab = vocab
        self.lex_extractor = lex_extractor
        self.max_seq_len = max_seq_len

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

        return item


class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.attn(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        return context


class BiLSTMStyleClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        lex_feature_dim: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2
        self.attn_pool = AttentionPooling(lstm_out_dim)

        self.lex_feature_dim = lex_feature_dim
        fusion_dim = lstm_out_dim + lex_feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        lex_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, T = token_ids.shape

        embedded = self.embedding(token_ids)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=T)

        mask = torch.arange(T, device=token_ids.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)

        context = self.attn_pool(lstm_out, mask)

        if lex_feats is not None and self.lex_feature_dim > 0:
            context = torch.cat([context, lex_feats], dim=-1)

        return self.classifier(context)


def load_data(config: Config):
    print(f"Loading from: {config.DATA_PATH}")

    if os.path.isdir(config.DATA_PATH):
        import glob

        csv_files = sorted(glob.glob(os.path.join(config.DATA_PATH, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in directory: {config.DATA_PATH}"
            )
        parts = []
        for f in csv_files:
            print(f"  reading: {f}")
            parts.append(pd.read_csv(f))
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(config.DATA_PATH)

    df = df.dropna(subset=[config.CODE_COLUMN, config.AUTHOR_COLUMN])
    df[config.CODE_COLUMN] = df[config.CODE_COLUMN].astype(str)

    counts = df[config.AUTHOR_COLUMN].value_counts()
    eligible = counts[counts >= config.MIN_SAMPLES_PER_AUTHOR]
    top_authors = eligible.head(config.TOP_N_AUTHORS).index.tolist()

    df = df[df[config.AUTHOR_COLUMN].isin(top_authors)].copy()
    author2idx = {a: i for i, a in enumerate(top_authors)}
    df["label"] = df[config.AUTHOR_COLUMN].map(author2idx)

    print(f"Authors  : {len(top_authors)}")
    print(f"Samples  : {len(df)}")
    print("Per-author sample counts:")
    for a in top_authors:
        print(f"  {a:<30s}  {(df[config.AUTHOR_COLUMN] == a).sum()}")

    return df, top_authors, author2idx


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in df["label"].unique():
        idx = df[df["label"] == label].index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_idx += idx[:n_test]
        val_idx += idx[n_test : n_test + n_val]
        train_idx += idx[n_test + n_val :]

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def make_collate_fn(use_lex: bool):
    def collate(batch: List[dict]) -> dict:
        token_ids = torch.stack([b["token_ids"] for b in batch])
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long).clamp(
            min=1
        )
        labels = torch.stack([b["label"] for b in batch])
        out = {"token_ids": token_ids, "lengths": lengths, "labels": labels}
        if use_lex:
            out["lex_feats"] = torch.stack([b["lex_feats"] for b in batch])
        return out

    return collate


def train_epoch(model, loader, optimizer, criterion, device, use_lex):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        optimizer.zero_grad()
        logits = model(token_ids, lengths, lex_feats)
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
        lex_feats = batch.get("lex_feats")
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        logits = model(token_ids, lengths, lex_feats)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(labels)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


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
    config = Config()

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print(f"Device : {config.DEVICE}")
    print(
        f"Authors: {config.TOP_N_AUTHORS}  |  Max seq len: {config.MAX_SEQ_LEN} chars\n"
    )

    df, top_authors, author2idx = load_data(config)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(
        f"\nSplit  → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}\n"
    )

    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    print(f"Vocabulary size : {len(vocab)}")

    lex_extractor = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    lex_dim = lex_extractor.feature_dim if lex_extractor else 0
    print(f"Lexical feat dim: {lex_dim}  (hand-crafted style signals)")

    def make_ds(split_df: pd.DataFrame) -> CodeStyleDataset:
        return CodeStyleDataset(
            codes=split_df[config.CODE_COLUMN].tolist(),
            labels=split_df["label"].tolist(),
            vocab=vocab,
            lex_extractor=lex_extractor,
            max_seq_len=config.MAX_SEQ_LEN,
        )

    collate_fn = make_collate_fn(config.USE_LEXICAL_FEATURES)
    train_loader = DataLoader(
        make_ds(train_df),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        make_ds(val_df),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        make_ds(test_df),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model = BiLSTMStyleClassifier(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=len(top_authors),
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        lex_feature_dim=lex_dim,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
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

    save_path = "bilstm_style_classifier.pt"
    torch.save(
        {
            "model_state": best_state,
            "vocab": vocab,
            "author2idx": author2idx,
            "lex_feature_dim": lex_dim,
            "config": config.__dict__,
        },
        save_path,
    )
    print(f"\nCheckpoint saved → {save_path}")


if __name__ == "__main__":
    main()
