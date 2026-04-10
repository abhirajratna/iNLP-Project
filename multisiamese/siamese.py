import os
import re
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SiameseConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 10  # Increased for meaningful pairs
    MAX_SEQ_LEN = 1000

    USE_LEXICAL_FEATURES = True
    VOCAB_SIZE = 200
    EMBED_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 5e-4  # Lower LR for siamese
    WEIGHT_DECAY = 1e-4
    MARGIN = 1.0  # Margin for contrastive loss
    SEED = 42

    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LexicalFeatureExtractor:
    KEYWORDS = [
        "for", "while", "if", "else", "switch", "return", "int", "long",
        "string", "vector", "auto", "void", "class", "struct", "typedef",
        "using", "namespace", "include", "define", "cout", "cin",
        "printf", "scanf", "break", "continue",
    ]

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

    def build(self, texts: List[str], max_vocab: int = 200) -> None:
        # Ensure all texts are strings to avoid TypeError when iterating
        str_texts = [str(t) for t in texts]
        counter = Counter(ch for text in str_texts for ch in text)
        most_common = [ch for ch, _ in counter.most_common(max_vocab - 2)]
        vocab = [self.PAD, self.UNK] + most_common
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        ids = [self.char2idx.get(ch, 1) for ch in text[:max_len]]
        length = len(ids)
        ids += [0] * (max_len - length)
        return ids, length

    def __len__(self) -> int:
        return len(self.char2idx)


class SiameseCodeDataset(Dataset):
    def __init__(
        self,
        codes: List[str],
        labels: List[int],
        vocab: CharVocabulary,
        lex_extractor: Optional[LexicalFeatureExtractor],
        max_seq_len: int,
        num_pairs: int = 5000
    ):
        self.codes = codes
        self.labels = labels
        self.vocab = vocab
        self.lex_extractor = lex_extractor
        self.max_seq_len = max_seq_len
        self.num_pairs = num_pairs

        # Group indices by author
        self.author_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.author_indices:
                self.author_indices[label] = []
            self.author_indices[label].append(idx)
        
        self.authors = list(self.author_indices.keys())

    def __len__(self) -> int:
        return self.num_pairs

    def _get_single_item(self, idx: int) -> dict:
        code = self.codes[idx]
        token_ids, length = self.vocab.encode(code, self.max_seq_len)
        item = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": length,
        }
        if self.lex_extractor:
            feats = self.lex_extractor.extract(code)
            item["lex_feats"] = torch.tensor(feats, dtype=torch.float)
        return item

    def __getitem__(self, _idx: int) -> dict:
        # 50% positive pairs, 50% negative pairs
        is_positive = random.random() > 0.5
        
        if is_positive:
            # Pick a random author who has at least 2 samples
            eligible_authors = [a for a in self.authors if len(self.author_indices[a]) >= 2]
            author = random.choice(eligible_authors)
            idx1, idx2 = random.sample(self.author_indices[author], 2)
            target = 1.0
        else:
            # Pick two different authors
            author1, author2 = random.sample(self.authors, 2)
            idx1 = random.choice(self.author_indices[author1])
            idx2 = random.choice(self.author_indices[author2])
            target = 0.0

        item1 = self._get_single_item(idx1)
        item2 = self._get_single_item(idx2)

        return {
            "x1": item1,
            "x2": item2,
            "target": torch.tensor(target, dtype=torch.float)
        }


def collate_siamese(batch: List[dict]) -> dict:
    def stack_items(items: List[dict]):
        out = {
            "token_ids": torch.stack([b["token_ids"] for b in items]),
            "lengths": torch.tensor([b["length"] for b in items], dtype=torch.long).clamp(min=1)
        }
        if "lex_feats" in items[0]:
            out["lex_feats"] = torch.stack([b["lex_feats"] for b in items])
        return out

    return {
        "x1": stack_items([b["x1"] for b in batch]),
        "x2": stack_items([b["x2"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch])
    }


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)


class SiameseCodeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        lex_feature_dim: int = 0,
        out_dim: int = 128
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
        
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, token_ids, lengths, lex_feats=None):
        _, T = token_ids.shape
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=T)
        mask = torch.arange(T, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        context = self.attn_pool(lstm_out, mask)
        if lex_feats is not None and self.lex_feature_dim > 0:
            context = torch.cat([context, lex_feats], dim=-1)
        return self.projection(context)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (target) * torch.pow(euclidean_distance, 2) +
            (1 - target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def load_raw_data(config: SiameseConfig):
    if os.path.isdir(config.DATA_PATH):
        import glob
        csv_files = sorted(glob.glob(os.path.join(config.DATA_PATH, "*.csv")))
        parts = [pd.read_csv(f) for f in csv_files]
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
    return df, author2idx


def train_siamese_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x1 = {k: v.to(device) for k, v in batch["x1"].items()}
        x2 = {k: v.to(device) for k, v in batch["x2"].items()}
        target = batch["target"].to(device)
        
        optimizer.zero_grad()
        out1 = model(**x1)
        out2 = model(**x2)
        loss = criterion(out1, out2, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_siamese(model, loader, device, margin=1.0):
    model.eval()
    dists, targets = [], []
    for batch in loader:
        x1 = {k: v.to(device) for k, v in batch["x1"].items()}
        x2 = {k: v.to(device) for k, v in batch["x2"].items()}
        target = batch["target"].to(device)
        out1 = model(**x1)
        out2 = model(**x2)
        dists.extend(F.pairwise_distance(out1, out2).cpu().tolist())
        targets.extend(target.cpu().tolist())
    
    dists = np.array(dists)
    targets = np.array(targets)
    # Simple accuracy based on margin/2 threshold
    preds = (dists < (margin / 2)).astype(float)
    acc = (preds == targets).mean()
    return acc


class VerificationSystem:
    def __init__(self, model, vocab, lex_extractor, device, threshold=0.5):
        self.model = model
        self.vocab = vocab
        self.lex_extractor = lex_extractor
        self.device = device
        self.threshold = threshold
        self.model.eval()

    def get_embedding(self, code: str) -> torch.Tensor:
        token_ids, length = self.vocab.encode(code, 1000)
        token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        lengths = torch.tensor([length], dtype=torch.long).to(self.device)
        
        lex_feats = None
        if self.lex_extractor:
            lex_feats = self.lex_extractor.extract(code)
            lex_feats = torch.tensor(lex_feats, dtype=torch.float).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(token_ids, lengths, lex_feats)
        return embedding

    def verify(self, reference_codes: List[str], query_code: str) -> Tuple[bool, float]:
        """
        Verify if the query_code was written by the same author as reference_codes.
        Uses averaged embeddings for comparison.
        """
        ref_embeddings = [self.get_embedding(c) for c in reference_codes]
        avg_ref_emb = torch.mean(torch.stack(ref_embeddings), dim=0)
        
        query_emb = self.get_embedding(query_code)
        
        dist = F.pairwise_distance(avg_ref_emb, query_emb).item()
        
        # If distance is small, it's the same author
        is_same = dist < self.threshold
        return is_same, dist


def main():
    config = SiameseConfig()
    torch.manual_seed(config.SEED)
    
    df, author2idx = load_raw_data(config)
    
    # Split
    authors = df["label"].unique()
    train_indices, val_indices, test_indices = [], [], []
    for a in authors:
        idx = df[df["label"] == a].index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_val, n_test = int(n * config.VAL_RATIO), int(n * config.TEST_RATIO)
        test_indices += idx[:n_test]
        val_indices += idx[n_test:n_test+n_val]
        train_indices += idx[n_test+n_val:]
    
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]

    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    lex = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    
    train_ds = SiameseCodeDataset(train_df[config.CODE_COLUMN].tolist(), train_df["label"].tolist(), vocab, lex, config.MAX_SEQ_LEN, num_pairs=5000)
    val_ds = SiameseCodeDataset(val_df[config.CODE_COLUMN].tolist(), val_df["label"].tolist(), vocab, lex, config.MAX_SEQ_LEN, num_pairs=1000)
    test_ds = SiameseCodeDataset(test_df[config.CODE_COLUMN].tolist(), test_df["label"].tolist(), vocab, lex, config.MAX_SEQ_LEN, num_pairs=1000)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_siamese)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_siamese)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_siamese)

    model = SiameseCodeEncoder(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        lex_feature_dim=lex.feature_dim if lex else 0,
        out_dim=128
    ).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = ContrastiveLoss(margin=config.MARGIN)

    print("Starting Siamese Training...")
    for epoch in range(1, config.EPOCHS + 1):
        loss = train_siamese_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_acc = evaluate_siamese(model, val_loader, config.DEVICE, margin=config.MARGIN)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    test_acc = evaluate_siamese(model, test_loader, config.DEVICE, margin=config.MARGIN)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Verification Demo
    print("\n--- Verification Demo ---")
    author_labels = list(set(test_df["label"]))
    author = author_labels[0]
    author_codes = test_df[test_df["label"] == author][config.CODE_COLUMN].tolist()
    
    if len(author_codes) >= 3:
        ref_codes = author_codes[:2]
        query_same = author_codes[2]
        
        diff_author = author_labels[1]
        query_diff = test_df[test_df["label"] == diff_author][config.CODE_COLUMN].iloc[0]
        
        verifier = VerificationSystem(model, vocab, lex, config.DEVICE, threshold=config.MARGIN/2)
        
        is_same1, dist1 = verifier.verify(ref_codes, query_same)
        print(f"Same Author Check: Expected True, Got {is_same1} (dist: {dist1:.4f})")
        
        is_same2, dist2 = verifier.verify(ref_codes, query_diff)
        print(f"Diff Author Check: Expected False, Got {is_same2} (dist: {dist2:.4f})")

    # Save
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'config': config,
        'lex_extractor': lex
    }, "siamese_author_model.pt")
    print("\nModel saved to siamese_author_model.pt")


if __name__ == "__main__":
    main()
