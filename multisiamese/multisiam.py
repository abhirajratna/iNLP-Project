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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MultiSiamConfig:
    DATA_PATH = "datasets/"
    CODE_COLUMN = "flines"
    AUTHOR_COLUMN = "username"
    TOP_N_AUTHORS = 20
    MIN_SAMPLES_PER_AUTHOR = 15
    MAX_SEQ_LEN = 1000

    USE_LEXICAL_FEATURES = False
    VOCAB_SIZE = 200
    EMBED_DIM = 64
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    OUT_DIM = 128

    GROUP_SIZE = 4  # Number of samples per author in one training "group"
    BATCH_SIZE = 8  # Batch of groups
    EPOCHS = 40
    LR = 5e-4
    MARGIN = 0.5  # Triplet Margin
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
        features += [sum(1 for l in lines if "//" in l) / n_lines, code.count("/*") / n_lines]
        lengths = [len(l) for l in lines if l.strip()]
        if lengths:
            features += [min(np.mean(lengths) / 80.0, 1.0), min(np.std(lengths) / 40.0, 1.0), min(max(lengths) / 120.0, 1.0)]
        else:
            features += [0.0, 0.0, 0.0]
        n_chars = max(len(code), 1)
        features += [code.count("{") / n_chars * 100, code.count(";") / n_chars * 100]
        features.append(sum(1 for l in lines if not l.strip()) / n_lines)
        features.append(sum(1 for t in ids if len(t) <= 2) / n_id)
        return features

    @property
    def feature_dim(self) -> int:
        return 3 + len(self.KEYWORDS) + 1 + 3 + 2 + 3 + 2 + 1 + 1


class CharVocabulary:
    PAD, UNK = "<PAD>", "<UNK>"
    def __init__(self): self.char2idx: Dict[str, int] = {}
    def build(self, texts: List[str], max_vocab: int = 200) -> None:
        str_texts = [str(t) for t in texts]
        counter = Counter(ch for text in str_texts for ch in text)
        vocab = [self.PAD, self.UNK] + [ch for ch, _ in counter.most_common(max_vocab - 2)]
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        ids = [self.char2idx.get(ch, 1) for ch in text[:max_len]]
        length = len(ids)
        return ids + [0] * (max_len - length), length
    def __len__(self) -> int: return len(self.char2idx)


class MultiAuthorDataset(Dataset):
    """Dataset that returns groups of snippets for each author to support MultiSiam triplet loss."""
    def __init__(self, codes: List[str], labels: List[int], vocab: CharVocabulary, lex: Optional[LexicalFeatureExtractor], config: MultiSiamConfig, num_groups: int = 2000):
        self.codes, self.labels, self.vocab, self.lex, self.config, self.num_groups = codes, labels, vocab, lex, config, num_groups
        self.author_indices = {}
        for idx, label in enumerate(labels):
            self.author_indices.setdefault(label, []).append(idx)
        self.authors = [a for a in self.author_indices.keys() if len(self.author_indices[a]) >= config.GROUP_SIZE]

    def _get_single_item(self, idx: int) -> dict:
        code = self.codes[idx]
        ids, length = self.vocab.encode(code, self.config.MAX_SEQ_LEN)
        item = {"token_ids": torch.tensor(ids, dtype=torch.long), "length": length}
        if self.lex: item["lex_feats"] = torch.tensor(self.lex.extract(code), dtype=torch.float)
        return item

    def __len__(self) -> int: return self.num_groups

    def __getitem__(self, _idx: int) -> dict:
        author = random.choice(self.authors)
        indices = random.sample(self.author_indices[author], self.config.GROUP_SIZE)
        return {"items": [self._get_single_item(i) for i in indices], "author_label": author}


def collate_multisiam(batch: List[dict]) -> dict:
    group_size = len(batch[0]["items"])
    batch_size = len(batch)
    all_items = [it for b in batch for it in b["items"]]
    out = {
        "token_ids": torch.stack([it["token_ids"] for it in all_items]),
        "lengths": torch.tensor([it["length"] for it in all_items], dtype=torch.long).clamp(min=1),
    }
    if "lex_feats" in all_items[0]:
        out["lex_feats"] = torch.stack([it["lex_feats"] for it in all_items])
    return {"data": out, "author_labels": torch.tensor([b["author_label"] for b in batch]), "group_size": group_size}


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class MultiSiamEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, lex_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn_pool = AttentionPooling(hidden_dim * 2)
        fusion_dim = (hidden_dim * 2) + lex_dim
        self.projection = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

    def forward(self, token_ids, lengths, lex_feats=None):
        _, T = token_ids.shape
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_packed, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_packed, batch_first=True, total_length=T)
        mask = torch.arange(T, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        context = self.attn_pool(lstm_out, mask)
        if lex_feats is not None: context = torch.cat([context, lex_feats], dim=-1)
        return F.normalize(self.projection(context), p=2, dim=-1)


class MultiSiamLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, group_size):
        B = embeddings.size(0) // group_size
        M = group_size
        
        # Avoid 0-vector collapse by strictly normalizing
        embs = F.normalize(embeddings, p=2, dim=-1)
        embs = embs.view(B, M, -1)
        
        loss = 0.0
        triplets_count = 0
        
        for i in range(B):
            for a_idx in range(M):
                anchor = embs[i, a_idx]
                
                # Sample a positive
                p_idx = (a_idx + 1) % M
                positive = embs[i, p_idx]
                
                # Sample a negative from a different group
                n_group = (i + 1) % B
                negative = embs[n_group, a_idx]
                
                loss += self.triplet_loss(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                triplets_count += 1
                
        return loss / max(triplets_count, 1)


class MultiSiamVerification:
    def __init__(self, model, vocab, lex, device, threshold=0.7):
        self.model, self.vocab, self.lex, self.device, self.threshold = model, vocab, lex, device, threshold
        self.model.eval()

    def get_embedding(self, codes: List[str]) -> torch.Tensor:
        all_ids, all_lens, all_lex = [], [], []
        for c in codes:
            ids, length = self.vocab.encode(c, 1000)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_lens.append(length)
            if self.lex: all_lex.append(torch.tensor(self.lex.extract(c), dtype=torch.float))
        
        ids_t = torch.stack(all_ids).to(self.device)
        lens_t = torch.tensor(all_lens).to(self.device)
        lex_t = torch.stack(all_lex).to(self.device) if all_lex else None
        
        with torch.no_grad():
            return self.model(ids_t, lens_t, lex_t)

    def verify(self, reference_codes: List[str], query_code: str) -> Tuple[bool, float]:
        """N-vs-1 Verification: Averaging reference embeddings to create an author signature."""
        ref_embs = self.get_embedding(reference_codes)
        ref_embs = F.normalize(ref_embs, p=2, dim=-1)
        author_signature = F.normalize(ref_embs.mean(dim=0, keepdim=True), p=2, dim=-1)
        
        query_emb = self.get_embedding([query_code])
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        
        dist = F.pairwise_distance(author_signature, query_emb).item()
        return dist < self.threshold, dist


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        data = {k: v.to(device) for k, v in batch["data"].items()}
        optimizer.zero_grad()
        out = model(**data)
        loss = criterion(out, batch["group_size"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_multisiam_metrics(model, vocab, lex, df, config, num_samples=500, threshold=None):
    """Evaluates the model on N-vs-1 verification tasks and returns standard metrics."""
    if threshold is None: threshold = 0.5
    verifier = MultiSiamVerification(model, vocab, lex, config.DEVICE, threshold=threshold)
    y_true, y_pred, distances = [], [], []
    
    authors = list(set(df["label"]))
    for _ in range(num_samples):
        is_same = random.random() > 0.5
        a1 = random.choice(authors)
        a1_codes = df[df["label"] == a1][config.CODE_COLUMN].tolist()
        
        if len(a1_codes) < config.GROUP_SIZE + 1: continue
        
        refs = random.sample(a1_codes, config.GROUP_SIZE)
        if is_same:
            query = random.choice([c for c in a1_codes if c not in refs])
            label = 1
        else:
            a2 = random.choice([a for a in authors if a != a1])
            query = random.choice(df[df["label"] == a2][config.CODE_COLUMN].tolist())
            label = 0
            
        is_match, dist = verifier.verify(refs, query)
        y_true.append(label)
        y_pred.append(int(is_match))
        distances.append(dist)
        
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "avg_dist": np.mean(distances)
    }
    return metrics, y_true, distances


def find_optimal_threshold(y_true, distances):
    """Finds the threshold that maximizes F1 score on the given distances."""
    best_f1 = 0
    best_thresh = 0.5
    # Since we use L2 distance, smaller is better (Match).
    # Test a range of thresholds
    thresholds = np.linspace(min(distances)-0.1, max(distances)+0.1, 50)
    for t in thresholds:
        y_pred = [1 if d < t else 0 for d in distances]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def main():
    cfg = MultiSiamConfig()
    torch.manual_seed(cfg.SEED)
    
    from siamese import load_raw_data
    df, author2idx = load_raw_data(cfg)
    authors = df["label"].unique()
    train_idx, val_idx, test_idx = [], [], []
    for a in authors:
        idx = df[df["label"] == a].index.tolist()
        random.shuffle(idx)
        n = len(idx)
        n_val, n_test = int(n * cfg.VAL_RATIO), int(n * cfg.TEST_RATIO)
        test_idx += idx[:n_test]; val_idx += idx[n_test:n_test+n_val]; train_idx += idx[n_test+n_val:]
    
    train_df, val_df, test_df = df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]
    vocab = CharVocabulary()
    vocab.build(train_df[cfg.CODE_COLUMN].tolist(), cfg.VOCAB_SIZE)
    lex = LexicalFeatureExtractor() if cfg.USE_LEXICAL_FEATURES else None
    
    train_ds = MultiAuthorDataset(train_df[cfg.CODE_COLUMN].tolist(), train_df["label"].tolist(), vocab, lex, cfg, num_groups=500)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_multisiam)
    
    model = MultiSiamEncoder(len(vocab), cfg.EMBED_DIM, cfg.HIDDEN_DIM, cfg.NUM_LAYERS, cfg.DROPOUT, lex.feature_dim if lex else 0, cfg.OUT_DIM).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = MultiSiamLoss(margin=cfg.MARGIN)
    
    print(f"Starting MultiSiam Training (Group Size: {cfg.GROUP_SIZE})...")
    best_val_f1 = 0
    best_thresh = 0.5
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE)
        val_metrics, y_v, d_v = evaluate_multisiam_metrics(model, vocab, lex, val_df, cfg, num_samples=200)
        
        # Track best model based on val performance
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_thresh, _ = find_optimal_threshold(y_v, d_v)
            torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'lex_extractor': lex, 'threshold': best_thresh}, "multisiam_best.pt")
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | NEW BEST! Val Acc: {val_metrics['accuracy']:.4f} | F1: {best_val_f1:.4f} | Thresh: {best_thresh:.3f}")
        else:
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

    print("\n--- Final Test Metrics (Using Best Model) ---")
    checkpoint = torch.load("multisiam_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    best_thresh = checkpoint['threshold']
    
    test_metrics, _, _ = evaluate_multisiam_metrics(model, vocab, lex, test_df, cfg, num_samples=500, threshold=best_thresh)
    for k, v in test_metrics.items():
        print(f"Test {k.capitalize()}: {v:.4f}")
    print(f"Using Optimal Threshold: {best_thresh:.4f}")

    # N-vs-1 Verification Demo
    print("\n--- N-vs-1 Verification Demo ---")
    verifier = MultiSiamVerification(model, vocab, lex, cfg.DEVICE, threshold=0.5)
    
    author_labels = list(set(test_df["label"]))
    if len(author_labels) >= 2:
        a1, a2 = author_labels[0], author_labels[1]
        a1_codes = test_df[test_df["label"] == a1][cfg.CODE_COLUMN].tolist()
        a2_codes = test_df[test_df["label"] == a2][cfg.CODE_COLUMN].tolist()
        
        if len(a1_codes) >= 5:
            refs = a1_codes[:4]
            q_pos = a1_codes[4]
            q_neg = a2_codes[0]
            
            is_same1, sim1 = verifier.verify(refs, q_pos)
            print(f"Same Author (N=4): Got {is_same1} (Similarity: {sim1:.4f})")
            
            is_same2, sim2 = verifier.verify(refs, q_neg)
            print(f"Diff Author (N=4): Got {is_same2} (Similarity: {sim2:.4f})")

    torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'lex_extractor': lex}, "multisiam_author_model.pt")
    print("\nMultiSiam model saved to multisiam_author_model.pt")

if __name__ == "__main__":
    main()
