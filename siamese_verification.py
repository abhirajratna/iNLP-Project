"""
Siamese Metric Learning for Code Authorship Verification
==========================================================
Goal: Given a person and their code samples, determine whether a new piece
      of code was written by them — a *verification* (binary) task rather
      than an *attribution* (multi-class) task.

Approach
--------
  1. **Embedding network** – Bi-LSTM + Attention Pooling (reused from
     sequential.py) followed by a projection MLP that maps code to a
     128-d L2-normalised embedding.
  2. **Metric losses** – combined Triplet + Contrastive loss with smart
     online mining (semi-hard triplets, hard negatives).
  3. **Class-balanced batching** – each batch contains samples from
     multiple authors with ≥2 samples per author, enabling rich in-batch
     pair / triplet enumeration.
  4. **Verification inference** – cosine similarity between two code
     embeddings, compared against a learned threshold (EER-optimal).

Dataset : GCJ (Google Code Jam C++ solutions)
Column  : flines   → raw source code
Label   : username → author to predict

Usage
-----
    python siamese_verification.py                     # full training
    python siamese_verification.py --epochs 2          # quick smoke test
"""

import os
import re
import math
import random
import argparse
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class SiameseConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    DATA_PATH              = "datasets/"
    CODE_COLUMN            = "flines"
    AUTHOR_COLUMN          = "username"
    TOP_N_AUTHORS          = 40
    MIN_SAMPLES_PER_AUTHOR = 5
    MAX_SEQ_LEN            = 2000

    # ── Vocabulary & lexical features ─────────────────────────────────────
    VOCAB_SIZE             = 200
    USE_LEXICAL_FEATURES   = True

    # ── Embedding network ─────────────────────────────────────────────────
    CHAR_EMBED_DIM         = 64
    LSTM_HIDDEN_DIM        = 256
    LSTM_NUM_LAYERS        = 2
    LSTM_DROPOUT           = 0.3
    PROJECTION_DIM         = 128    # final embedding dimension

    # ── Metric learning ───────────────────────────────────────────────────
    MARGIN_TRIPLET         = 1.0
    MARGIN_CONTRASTIVE     = 1.0
    LOSS_WEIGHT_TRIPLET    = 0.6    # α  (contrastive gets 1 − α)

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE             = 64     # larger batches → richer in-batch mining
    AUTHORS_PER_BATCH      = 8     # P  (class-balanced: P authors × K each)
    SAMPLES_PER_AUTHOR     = 8     # K
    EPOCHS                 = 30
    WARMUP_EPOCHS          = 3     # freeze LSTM, train projection only
    LR                     = 1e-3
    LR_FINETUNE            = 3e-4
    WEIGHT_DECAY           = 1e-4
    SEED                   = 42

    # ── Split ratios ──────────────────────────────────────────────────────
    VAL_RATIO              = 0.10
    TEST_RATIO             = 0.10

    # ── Paths ─────────────────────────────────────────────────────────────
    PRETRAINED_PATH        = "bilstm_style_classifier.pt"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Lexical Feature Extractor  (copied from sequential.py for self-containment)
# ─────────────────────────────────────────────────────────────────────────────

class LexicalFeatureExtractor:
    KEYWORDS = [
        'for', 'while', 'if', 'else', 'switch', 'return',
        'int', 'long', 'string', 'vector', 'auto', 'void',
        'class', 'struct', 'typedef', 'using', 'namespace',
        'include', 'define', 'cout', 'cin', 'printf', 'scanf',
        'break', 'continue',
    ]

    def __call__(self, code: str) -> List[float]:
        return self.extract(code)

    def extract(self, code: str) -> List[float]:
        lines  = code.split('\n')
        tokens = re.findall(r'\b\w+\b', code)
        ids    = [t for t in tokens if not t.isdigit() and t not in self.KEYWORDS]

        features: List[float] = []
        n_id = max(len(ids), 1)

        # Naming conventions
        camel    = sum(1 for t in ids if re.search(r'[a-z][A-Z]', t))
        snake    = sum(1 for t in ids if '_' in t and t != '_')
        all_caps = sum(1 for t in ids if t.isupper() and len(t) > 1)
        features += [camel / n_id, snake / n_id, all_caps / n_id]

        # Keyword frequencies
        tok_counts  = Counter(tokens)
        total_toks  = max(len(tokens), 1)
        for kw in self.KEYWORDS:
            features.append(tok_counts.get(kw, 0) / total_toks)

        # For-vs-while ratio
        n_for   = tok_counts.get('for', 0)
        n_while = tok_counts.get('while', 0)
        features.append(n_for / max(n_for + n_while, 1))

        # Indentation style
        indented = [l for l in lines if l and l[0] in (' ', '\t')]
        n_ind    = max(len(indented), 1)
        tab_rate   = sum(1 for l in indented if l[0] == '\t') / n_ind
        space_rate = sum(1 for l in indented if l[0] == ' ')  / n_ind
        depths = [len(l) - len(l.lstrip(' ')) for l in lines if l.startswith(' ')]
        avg_indent = (np.mean(depths) / 8.0) if depths else 0.0
        features += [tab_rate, space_rate, min(avg_indent, 1.0)]

        # Comment style
        n_lines = max(len(lines), 1)
        features += [
            sum(1 for l in lines if '//' in l) / n_lines,
            code.count('/*') / n_lines,
        ]

        # Line-length stats
        lengths = [len(l) for l in lines if l.strip()]
        if lengths:
            features += [
                min(np.mean(lengths) / 80.0, 1.0),
                min(np.std(lengths)  / 40.0, 1.0),
                min(max(lengths)     / 120.0, 1.0),
            ]
        else:
            features += [0.0, 0.0, 0.0]

        # Syntactic density
        n_chars = max(len(code), 1)
        features += [
            code.count('{') / n_chars * 100,
            code.count(';') / n_chars * 100,
        ]

        # Blank-line ratio
        features.append(sum(1 for l in lines if not l.strip()) / n_lines)

        # Short-identifier ratio
        features.append(sum(1 for t in ids if len(t) <= 2) / n_id)

        return features

    @property
    def feature_dim(self) -> int:
        return 3 + len(self.KEYWORDS) + 1 + 3 + 2 + 3 + 2 + 1 + 1


# ─────────────────────────────────────────────────────────────────────────────
# Character-Level Vocabulary  (from sequential.py)
# ─────────────────────────────────────────────────────────────────────────────

class CharVocabulary:
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

    def build(self, texts: List[str], max_vocab: int = 200) -> None:
        counter   = Counter(ch for text in texts for ch in text)
        most_common = [ch for ch, _ in counter.most_common(max_vocab - 2)]
        vocab     = [self.PAD, self.UNK] + most_common
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        ids    = [self.char2idx.get(ch, 1) for ch in text[:max_len]]
        length = len(ids)
        ids   += [0] * (max_len - length)
        return ids, length

    def __len__(self) -> int:
        return len(self.char2idx)


# ─────────────────────────────────────────────────────────────────────────────
# Attention Pooling  (from sequential.py)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(outputs).squeeze(-1)
        scores  = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        context = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        return context


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Network
# ─────────────────────────────────────────────────────────────────────────────

class CodeEmbeddingNet(nn.Module):
    """
    Bi-LSTM + Attention Pooling → Projection MLP → L2-normalised embedding.

    The backbone (embedding + LSTM + attention) is structurally identical to
    BiLSTMStyleClassifier from sequential.py, but the classifier head is
    replaced with a metric-learning projection head.
    """

    def __init__(
        self,
        vocab_size:      int,
        embed_dim:       int   = 64,
        hidden_dim:      int   = 256,
        num_layers:      int   = 2,
        dropout:         float = 0.3,
        lex_feature_dim: int   = 0,
        projection_dim:  int   = 128,
    ):
        super().__init__()

        self.lex_feature_dim = lex_feature_dim

        # ── Backbone (same as BiLSTMStyleClassifier) ──────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2
        self.attn_pool = AttentionPooling(lstm_out_dim)

        fusion_dim = lstm_out_dim + lex_feature_dim

        # ── Projection head (replaces classifier) ─────────────────────────
        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, projection_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
        lex_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns L2-normalised embeddings [B, projection_dim]."""
        _, T = token_ids.shape

        embedded = self.embedding(token_ids)

        packed     = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_packed, _ = self.lstm(packed)
        lstm_out, _    = pad_packed_sequence(
            lstm_packed, batch_first=True, total_length=T
        )

        mask    = (
            torch.arange(T, device=token_ids.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        context = self.attn_pool(lstm_out, mask)

        if lex_feats is not None and self.lex_feature_dim > 0:
            context = torch.cat([context, lex_feats], dim=-1)

        proj = self.projector(context)
        return F.normalize(proj, p=2, dim=-1)

    def load_pretrained_backbone(self, checkpoint_path: str, device: str) -> None:
        """Load weights from a trained BiLSTMStyleClassifier checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠ Pretrained checkpoint not found: {checkpoint_path}")
            return

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state', ckpt)

        # Map classifier-model keys → embedding-model keys
        loaded = 0
        own_state = self.state_dict()
        for key, value in state.items():
            # Skip classifier head weights — we have our own projector
            if key.startswith('classifier.'):
                continue
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                loaded += 1

        self.load_state_dict(own_state)
        print(f"  ✓ Loaded {loaded} parameter tensors from pretrained checkpoint")


# ─────────────────────────────────────────────────────────────────────────────
# Smart Data Selection
# ─────────────────────────────────────────────────────────────────────────────

class MetricLearningDataset(Dataset):
    """
    Base dataset that stores code samples indexed by author for efficient
    mining.  Each __getitem__ returns a single encoded sample with its label.
    """

    def __init__(
        self,
        codes:         List[str],
        labels:        List[int],
        vocab:         CharVocabulary,
        lex_extractor: Optional[LexicalFeatureExtractor],
        max_seq_len:   int,
    ):
        self.codes         = codes
        self.labels        = labels
        self.vocab         = vocab
        self.lex_extractor = lex_extractor
        self.max_seq_len   = max_seq_len

        # Build per-author index for fast mining
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)
        self.unique_labels = sorted(self.label_to_indices.keys())

    def __len__(self) -> int:
        return len(self.codes)

    def _encode(self, idx: int) -> dict:
        code = self.codes[idx]
        token_ids, length = self.vocab.encode(code, self.max_seq_len)
        item = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'length':    length,
            'label':     torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.lex_extractor is not None:
            feats = self.lex_extractor.extract(code)
            item['lex_feats'] = torch.tensor(feats, dtype=torch.float)
        return item

    def __getitem__(self, idx: int) -> dict:
        return self._encode(idx)


class ClassBalancedBatchSampler(Sampler):
    """
    Yields batches of P × K indices: P randomly chosen authors, K randomly
    chosen samples per author.  This ensures every batch has rich same-author
    and different-author pairs for in-batch mining.

    If an author has fewer than K samples, we sample with replacement.
    """

    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        authors_per_batch: int,
        samples_per_author: int,
        num_batches: int,
    ):
        self.label_to_indices  = label_to_indices
        self.P                 = authors_per_batch
        self.K                 = samples_per_author
        self.num_batches       = num_batches
        self.unique_labels     = list(label_to_indices.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            # Choose P authors
            chosen = random.sample(
                self.unique_labels, min(self.P, len(self.unique_labels))
            )
            batch = []
            for lbl in chosen:
                pool = self.label_to_indices[lbl]
                if len(pool) >= self.K:
                    selected = random.sample(pool, self.K)
                else:
                    selected = random.choices(pool, k=self.K)
                batch.extend(selected)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class VerificationPairDataset(Dataset):
    """
    For evaluation: yields balanced positive (same-author) and negative
    (different-author) pairs with a binary label.
    """

    def __init__(
        self,
        codes:         List[str],
        labels:        List[int],
        vocab:         CharVocabulary,
        lex_extractor: Optional[LexicalFeatureExtractor],
        max_seq_len:   int,
        num_pairs:     int = 2000,
        seed:          int = 42,
    ):
        self.codes         = codes
        self.labels        = labels
        self.vocab         = vocab
        self.lex_extractor = lex_extractor
        self.max_seq_len   = max_seq_len

        rng = random.Random(seed)
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            label_to_indices[lbl].append(idx)

        # Authors with ≥2 samples (needed for positive pairs)
        eligible = {k: v for k, v in label_to_indices.items() if len(v) >= 2}
        eligible_labels = list(eligible.keys())

        self.pairs: List[Tuple[int, int, int]] = []   # (idx_a, idx_b, same?)

        n_pos = num_pairs // 2
        n_neg = num_pairs - n_pos

        # ── Positive pairs ────────────────────────────────────────────────
        for _ in range(n_pos):
            lbl = rng.choice(eligible_labels)
            a, b = rng.sample(eligible[lbl], 2)
            self.pairs.append((a, b, 1))

        # ── Negative pairs (hard: from similar-sized author pools) ────────
        for _ in range(n_neg):
            l1, l2 = rng.sample(eligible_labels, 2)
            a = rng.choice(eligible[l1])
            b = rng.choice(eligible[l2])
            self.pairs.append((a, b, 0))

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_idx(self, idx: int) -> dict:
        code = self.codes[idx]
        token_ids, length = self.vocab.encode(code, self.max_seq_len)
        item = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'length':    length,
        }
        if self.lex_extractor is not None:
            feats = self.lex_extractor.extract(code)
            item['lex_feats'] = torch.tensor(feats, dtype=torch.float)
        return item

    def __getitem__(self, idx: int) -> dict:
        i, j, same = self.pairs[idx]
        a = self._encode_idx(i)
        b = self._encode_idx(j)
        return {
            'a_token_ids':  a['token_ids'],
            'a_length':     a['length'],
            'b_token_ids':  b['token_ids'],
            'b_length':     b['length'],
            'a_lex_feats':  a.get('lex_feats'),
            'b_lex_feats':  b.get('lex_feats'),
            'same_author':  torch.tensor(same, dtype=torch.float),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions with Online Mining
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix [B, B].
    Distance = 1 - cosine_similarity, range [0, 2].
    Natural for L2-normalised embeddings and numerically stable.
    """
    # For L2-normalised vectors: cosine_sim = dot product
    sim = embeddings @ embeddings.t()                     # [B, B]
    sim = torch.clamp(sim, -1.0, 1.0)                    # numerical safety
    D   = 1.0 - sim                                       # [B, B] in [0, 2]
    D   = torch.clamp(D, min=0.0)                         # ensure non-negative
    return D


def online_triplet_loss(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    margin:     float = 1.0,
) -> Tuple[torch.Tensor, int]:
    """
    Batch-all triplet loss with semi-hard mining.

    For every valid (anchor, positive, negative) triplet in the batch:
      - Positive: same label as anchor
      - Negative: different label from anchor
      - Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin

    Falls back to hard negatives if no semi-hard triplets exist.
    """
    D = pairwise_distances(embeddings)
    B = embeddings.size(0)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # [B, B]

    loss  = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(B):
        # Positive indices (same label, not self)
        pos_mask = labels_eq[i].clone()
        pos_mask[i] = False
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]

        # Negative indices (different label)
        neg_mask    = ~labels_eq[i]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        for p in pos_indices:
            d_ap = D[i, p]

            # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
            d_an = D[i, neg_indices]
            semi_hard = (d_an > d_ap) & (d_an < d_ap + margin)

            if semi_hard.any():
                # Pick the hardest among semi-hard (closest to anchor)
                sh_dists = d_an[semi_hard]
                hardest_sh = sh_dists.min()
                triplet_loss = F.relu(d_ap - hardest_sh + margin)
            else:
                # Fall back to hardest negative overall
                hardest_neg = d_an.min()
                triplet_loss = F.relu(d_ap - hardest_neg + margin)

            loss  += triplet_loss
            count += 1

    if count > 0:
        loss = loss / count
    else:
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss, count


def online_contrastive_loss(
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    margin:     float = 1.0,
) -> Tuple[torch.Tensor, int]:
    """
    Batch-all contrastive loss with hard pair mining.

    For positive pairs (same label): penalise large distances.
    For negative pairs (different label): penalise distances < margin.
    Only considers the hardest pairs per sample for efficiency.
    """
    D = pairwise_distances(embeddings)
    B = embeddings.size(0)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

    loss  = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(B):
        # Hardest positive: same label, farthest away
        pos_mask = labels_eq[i].clone()
        pos_mask[i] = False
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]

        neg_mask    = ~labels_eq[i]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        if len(pos_indices) > 0:
            hardest_pos_dist = D[i, pos_indices].max()
            loss  += hardest_pos_dist ** 2
            count += 1

        if len(neg_indices) > 0:
            hardest_neg_dist = D[i, neg_indices].min()
            neg_loss = F.relu(margin - hardest_neg_dist) ** 2
            loss  += neg_loss
            count += 1

    if count > 0:
        loss = loss / count
    else:
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss, count


# ─────────────────────────────────────────────────────────────────────────────
# Collate Functions
# ─────────────────────────────────────────────────────────────────────────────

def metric_collate(batch: List[dict], use_lex: bool) -> dict:
    """Collate for MetricLearningDataset (single samples with labels)."""
    token_ids = torch.stack([b['token_ids'] for b in batch])
    lengths   = torch.tensor(
        [b['length'] for b in batch], dtype=torch.long
    ).clamp(min=1)
    labels    = torch.stack([b['label'] for b in batch])
    out = {'token_ids': token_ids, 'lengths': lengths, 'labels': labels}
    if use_lex:
        out['lex_feats'] = torch.stack([b['lex_feats'] for b in batch])
    return out


def pair_collate(batch: List[dict], use_lex: bool) -> dict:
    """Collate for VerificationPairDataset."""
    a_ids = torch.stack([b['a_token_ids'] for b in batch])
    a_len = torch.tensor(
        [b['a_length'] for b in batch], dtype=torch.long
    ).clamp(min=1)
    b_ids = torch.stack([b['b_token_ids'] for b in batch])
    b_len = torch.tensor(
        [b['b_length'] for b in batch], dtype=torch.long
    ).clamp(min=1)
    same  = torch.stack([b['same_author'] for b in batch])

    out = {
        'a_token_ids': a_ids, 'a_lengths': a_len,
        'b_token_ids': b_ids, 'b_lengths': b_len,
        'same_author': same,
    }
    if use_lex:
        out['a_lex_feats'] = torch.stack([b['a_lex_feats'] for b in batch])
        out['b_lex_feats'] = torch.stack([b['b_lex_feats'] for b in batch])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading  (from sequential.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: SiameseConfig):
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
    df['label'] = df[config.AUTHOR_COLUMN].map(author2idx)

    print(f"Authors  : {len(top_authors)}")
    print(f"Samples  : {len(df)}")
    for a in top_authors:
        print(f"  {a:<30s}  {(df[config.AUTHOR_COLUMN] == a).sum()}")

    return df, top_authors, author2idx


def stratified_split(
    df: pd.DataFrame, seed: int,
    val_ratio: float = 0.10, test_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label in df['label'].unique():
        idx = df[df['label'] == label].index.tolist()
        rng.shuffle(idx)
        n       = len(idx)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        test_idx  += idx[:n_test]
        val_idx   += idx[n_test: n_test + n_val]
        train_idx += idx[n_test + n_val:]

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx  ].reset_index(drop=True),
        df.loc[test_idx ].reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(
    similarities: np.ndarray,
    labels:       np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    EER is the point where FAR == FRR on the DET curve.
    As threshold increases from -1 → 1:
      - FAR (false accept rate) DECREASES  (fewer imposters accepted)
      - FRR (false reject rate) INCREASES  (more genuine pairs rejected)
    EER is where they cross.

    Args:
        similarities: cosine similarities between pairs
        labels:       1 for same-author, 0 for different-author

    Returns:
        (eer, threshold)
    """
    # Sweep thresholds from LOW to HIGH
    thresholds = np.linspace(-1.0, 1.0, 2000)
    pos_mask = labels == 1
    neg_mask = labels == 0

    fars = np.zeros(len(thresholds))
    frrs = np.zeros(len(thresholds))

    for i, thr in enumerate(thresholds):
        preds = (similarities >= thr).astype(int)
        # FAR: fraction of different-author pairs INCORRECTLY predicted as same
        fars[i] = preds[neg_mask].mean() if neg_mask.any() else 0.0
        # FRR: fraction of same-author pairs INCORRECTLY predicted as different
        frrs[i] = 1.0 - preds[pos_mask].mean() if pos_mask.any() else 0.0

    # Find the crossing point: where FAR drops below FRR
    # At low threshold: FAR ≈ 1, FRR ≈ 0  →  diff = FAR - FRR > 0
    # At high threshold: FAR ≈ 0, FRR ≈ 1  →  diff = FAR - FRR < 0
    diffs = fars - frrs

    # Find sign change
    for i in range(1, len(diffs)):
        if diffs[i - 1] >= 0 and diffs[i] < 0:
            # Linear interpolation for the crossing point
            w = diffs[i - 1] / (diffs[i - 1] - diffs[i])
            eer_val = float(fars[i - 1] * (1 - w) + fars[i] * w)
            thr_val = float(thresholds[i - 1] * (1 - w) + thresholds[i] * w)
            return eer_val, thr_val

    # Fallback: find the threshold with minimum |FAR - FRR|
    idx = int(np.argmin(np.abs(diffs)))
    eer_val = float((fars[idx] + frrs[idx]) / 2.0)
    return eer_val, float(thresholds[idx])


def compute_auc(
    similarities: np.ndarray,
    labels:       np.ndarray,
) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn needed)."""
    thresholds = np.linspace(1.0, -1.0, 2000)
    tpr_list, fpr_list = [], []

    for thr in thresholds:
        preds = (similarities >= thr).astype(int)
        pos_mask = labels == 1
        neg_mask = labels == 0

        tpr = preds[pos_mask].mean() if pos_mask.any() else 0.0
        fpr = preds[neg_mask].mean() if neg_mask.any() else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0

    return abs(auc)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     CodeEmbeddingNet,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    config:    SiameseConfig,
    device:    str,
    use_lex:   bool,
) -> Tuple[float, float, float]:
    """
    One training epoch with online triplet + contrastive loss.
    Returns (combined_loss, triplet_loss, contrastive_loss).
    """
    model.train()
    total_loss, total_tri, total_con = 0.0, 0.0, 0.0
    n_batches = 0

    for batch in loader:
        token_ids = batch['token_ids'].to(device)
        lengths   = batch['lengths'].to(device)
        labels    = batch['labels'].to(device)
        lex_feats = batch.get('lex_feats')
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        optimizer.zero_grad()

        embeddings = model(token_ids, lengths, lex_feats)

        # ── Triplet loss (online semi-hard mining) ────────────────────
        tri_loss, tri_count = online_triplet_loss(
            embeddings, labels, margin=config.MARGIN_TRIPLET
        )

        # ── Contrastive loss (online hard mining) ─────────────────────
        con_loss, con_count = online_contrastive_loss(
            embeddings, labels, margin=config.MARGIN_CONTRASTIVE
        )

        # ── Combined ──────────────────────────────────────────────────
        alpha = config.LOSS_WEIGHT_TRIPLET
        loss  = alpha * tri_loss + (1 - alpha) * con_loss

        if loss.requires_grad:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_tri  += tri_loss.item()
        total_con  += con_loss.item()
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_tri / n, total_con / n


@torch.no_grad()
def evaluate_verification(
    model:   CodeEmbeddingNet,
    loader:  DataLoader,
    device:  str,
    use_lex: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities and labels for all verification pairs.
    Returns (similarities, labels) as numpy arrays.
    """
    model.eval()
    all_sims   = []
    all_labels = []

    for batch in loader:
        a_ids = batch['a_token_ids'].to(device)
        a_len = batch['a_lengths'].to(device)
        b_ids = batch['b_token_ids'].to(device)
        b_len = batch['b_lengths'].to(device)
        same  = batch['same_author']

        a_lex = batch.get('a_lex_feats')
        b_lex = batch.get('b_lex_feats')
        if a_lex is not None:
            a_lex = a_lex.to(device)
            b_lex = b_lex.to(device)

        emb_a = model(a_ids, a_len, a_lex)
        emb_b = model(b_ids, b_len, b_lex)

        sims = F.cosine_similarity(emb_a, emb_b, dim=-1)
        all_sims.extend(sims.cpu().tolist())
        all_labels.extend(same.tolist())

    return np.array(all_sims), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# PCA Embedding Visualization
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_embeddings(
    model:  CodeEmbeddingNet,
    ds:     MetricLearningDataset,
    config: SiameseConfig,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings for all samples in a dataset."""
    model.eval()
    loader = DataLoader(
        ds,
        batch_size = 64,
        shuffle    = False,
        collate_fn = lambda b: metric_collate(b, config.USE_LEXICAL_FEATURES),
        num_workers= 0,
    )

    all_embs   = []
    all_labels = []

    for batch in loader:
        token_ids = batch['token_ids'].to(device)
        lengths   = batch['lengths'].to(device)
        lex_feats = batch.get('lex_feats')
        if lex_feats is not None:
            lex_feats = lex_feats.to(device)

        embs = model(token_ids, lengths, lex_feats)
        all_embs.append(embs.cpu().numpy())
        all_labels.append(batch['labels'].numpy())

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Apply PCA to reduce embeddings to 2D (no sklearn dependency)."""
    # Center the data
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Covariance matrix
    cov = np.cov(centered, rowvar=False)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    top_2 = eigenvectors[:, idx[:2]]

    # Project
    return centered @ top_2


def visualize_embeddings(
    model:      CodeEmbeddingNet,
    ds:         MetricLearningDataset,
    config:     SiameseConfig,
    author2idx: Dict[str, int],
    tag:        str,
    device:     str,
    max_samples: int = 1500,
) -> None:
    """
    Extract embeddings, apply PCA, and save a scatter plot colored by author.
    """
    print(f"  Extracting embeddings ({tag}) ...")
    embs, labels = extract_all_embeddings(model, ds, config, device)

    # Subsample if too many for clear visualization
    if len(embs) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(embs), max_samples, replace=False)
        embs   = embs[idx]
        labels = labels[idx]

    print(f"  Running PCA on {len(embs)} embeddings ...")
    coords = pca_2d(embs)

    idx2author = {v: k for k, v in author2idx.items()}
    unique_labels = sorted(set(labels.tolist()))
    n_authors = len(unique_labels)

    # Generate a nice colormap
    cmap = plt.cm.get_cmap('tab20', n_authors)

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = idx2author.get(lbl, f"Author {lbl}")
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=name, alpha=0.6, s=25, edgecolors='none',
        )

    ax.set_title(f'Code Embeddings — PCA 2D ({tag})', fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.legend(
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=8, markerscale=2, frameon=True,
        title='Authors', title_fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_path = f'plots/pca_embeddings_{tag}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Siamese Metric Learning for Code Authorship Verification")
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()

    config = SiameseConfig()
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 65)
    print("  Siamese Metric Learning — Code Authorship Verification")
    print("=" * 65)
    print(f"Device      : {config.DEVICE}")
    print(f"Authors     : {config.TOP_N_AUTHORS}")
    print(f"Seq len     : {config.MAX_SEQ_LEN} chars")
    print(f"Embedding   : {config.PROJECTION_DIM}-d")
    print(f"Margin (tri): {config.MARGIN_TRIPLET}  Margin (con): {config.MARGIN_CONTRASTIVE}")
    print(f"Loss mix    : α={config.LOSS_WEIGHT_TRIPLET} triplet + "
          f"{1-config.LOSS_WEIGHT_TRIPLET} contrastive\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    df, top_authors, author2idx = load_data(config)
    train_df, val_df, test_df = stratified_split(
        df, config.SEED, config.VAL_RATIO, config.TEST_RATIO
    )
    print(f"\nSplit → train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}\n")

    # ── 2. Vocabulary ─────────────────────────────────────────────────────
    vocab = CharVocabulary()
    vocab.build(train_df[config.CODE_COLUMN].tolist(), max_vocab=config.VOCAB_SIZE)
    print(f"Vocabulary size    : {len(vocab)}")

    lex_extractor = LexicalFeatureExtractor() if config.USE_LEXICAL_FEATURES else None
    lex_dim       = lex_extractor.feature_dim if lex_extractor else 0
    print(f"Lexical feat dim   : {lex_dim}")

    # ── 3. Datasets ───────────────────────────────────────────────────────
    train_ds = MetricLearningDataset(
        codes         = train_df[config.CODE_COLUMN].tolist(),
        labels        = train_df['label'].tolist(),
        vocab         = vocab,
        lex_extractor = lex_extractor,
        max_seq_len   = config.MAX_SEQ_LEN,
    )

    # Class-balanced batch sampler for training
    n_train_batches = max(len(train_ds) // config.BATCH_SIZE, 20)
    batch_sampler = ClassBalancedBatchSampler(
        label_to_indices  = train_ds.label_to_indices,
        authors_per_batch = config.AUTHORS_PER_BATCH,
        samples_per_author= config.SAMPLES_PER_AUTHOR,
        num_batches       = n_train_batches,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler = batch_sampler,
        collate_fn    = lambda b: metric_collate(b, config.USE_LEXICAL_FEATURES),
        num_workers   = 2,
    )

    # Verification pair datasets for val and test
    val_pair_ds = VerificationPairDataset(
        codes         = val_df[config.CODE_COLUMN].tolist(),
        labels        = val_df['label'].tolist(),
        vocab         = vocab,
        lex_extractor = lex_extractor,
        max_seq_len   = config.MAX_SEQ_LEN,
        num_pairs     = min(2000, len(val_df) * 5),
        seed          = config.SEED,
    )
    val_loader = DataLoader(
        val_pair_ds,
        batch_size = 32,
        shuffle    = False,
        collate_fn = lambda b: pair_collate(b, config.USE_LEXICAL_FEATURES),
        num_workers= 2,
    )

    test_pair_ds = VerificationPairDataset(
        codes         = test_df[config.CODE_COLUMN].tolist(),
        labels        = test_df['label'].tolist(),
        vocab         = vocab,
        lex_extractor = lex_extractor,
        max_seq_len   = config.MAX_SEQ_LEN,
        num_pairs     = min(3000, len(test_df) * 5),
        seed          = config.SEED + 1,
    )
    test_loader = DataLoader(
        test_pair_ds,
        batch_size = 32,
        shuffle    = False,
        collate_fn = lambda b: pair_collate(b, config.USE_LEXICAL_FEATURES),
        num_workers= 2,
    )

    # ── 4. Model ──────────────────────────────────────────────────────────
    model = CodeEmbeddingNet(
        vocab_size      = len(vocab),
        embed_dim       = config.CHAR_EMBED_DIM,
        hidden_dim      = config.LSTM_HIDDEN_DIM,
        num_layers      = config.LSTM_NUM_LAYERS,
        dropout         = config.LSTM_DROPOUT,
        lex_feature_dim = lex_dim,
        projection_dim  = config.PROJECTION_DIM,
    ).to(config.DEVICE)

    # Try loading pretrained backbone
    print(f"\nAttempting to load pretrained backbone from: {config.PRETRAINED_PATH}")
    model.load_pretrained_backbone(config.PRETRAINED_PATH, config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters   : {n_params:,}\n")
    print(model)

    # ── PCA visualization BEFORE training ─────────────────────────────────
    print("\n─── PCA Visualization: Before Training ─────────────────────────")
    visualize_embeddings(
        model      = model,
        ds         = train_ds,
        config     = config,
        author2idx = author2idx,
        tag        = "before_training",
        device     = config.DEVICE,
    )

    # ── 5. Two-phase training ─────────────────────────────────────────────
    best_eer       = 1.0
    best_threshold = 0.5
    best_state     = None

    # ── Phase 1: Warm-up (freeze backbone, train projector only) ──────
    warmup_epochs = min(config.WARMUP_EPOCHS, config.EPOCHS)

    if warmup_epochs > 0 and os.path.exists(config.PRETRAINED_PATH):
        print("\n─── Phase 1: Warm-up (projector only) ────────────────────────")

        for name, param in model.named_parameters():
            if not name.startswith('projector'):
                param.requires_grad = False

        opt_warmup = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LR,
        )

        for epoch in range(1, warmup_epochs + 1):
            tr_loss, tr_tri, tr_con = train_epoch(
                model, train_loader, opt_warmup, config,
                config.DEVICE, config.USE_LEXICAL_FEATURES,
            )
            sims, labels = evaluate_verification(
                model, val_loader, config.DEVICE, config.USE_LEXICAL_FEATURES,
            )
            eer, thr = compute_eer(sims, labels)

            marker = " ◀" if eer < best_eer else ""
            if eer < best_eer:
                best_eer       = eer
                best_threshold = thr
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"  WU {epoch}/{warmup_epochs}  "
                  f"Loss: {tr_loss:.4f} (T:{tr_tri:.4f} C:{tr_con:.4f})  "
                  f"EER: {eer:.4f}  Thr: {thr:.3f}{marker}")

        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True

    # ── Phase 2: Full fine-tuning ─────────────────────────────────────
    remaining = config.EPOCHS - warmup_epochs if os.path.exists(config.PRETRAINED_PATH) else config.EPOCHS
    print(f"\n─── Phase 2: Full training ({remaining} epochs) ──────────────────")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR_FINETUNE if os.path.exists(config.PRETRAINED_PATH) else config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=1e-6,
    )

    patience     = 7
    patience_ctr = 0

    col = 78
    print(f"\n{'Ep':>3}  {'Loss':>8}  {'Triplet':>8}  {'Contras':>8}  "
          f"{'EER':>6}  {'Thr':>5}  {'AUC':>6}")
    print("─" * col)

    for epoch in range(1, remaining + 1):
        tr_loss, tr_tri, tr_con = train_epoch(
            model, train_loader, optimizer, config,
            config.DEVICE, config.USE_LEXICAL_FEATURES,
        )
        scheduler.step()

        sims, labels = evaluate_verification(
            model, val_loader, config.DEVICE, config.USE_LEXICAL_FEATURES,
        )
        eer, thr = compute_eer(sims, labels)
        auc      = compute_auc(sims, labels)

        marker = " ◀" if eer < best_eer else ""
        if eer < best_eer:
            best_eer       = eer
            best_threshold = thr
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr   = 0
        else:
            patience_ctr += 1

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_tri:>8.4f}  {tr_con:>8.4f}  "
              f"{eer:>6.4f}  {thr:>5.3f}  {auc:>6.4f}{marker}")

        if patience_ctr >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    # ── 6. Restore best and evaluate on test ──────────────────────────────
    print(f"\nBest validation EER  : {best_eer:.4f}")
    print(f"Best threshold       : {best_threshold:.3f}")

    if best_state is not None:
        print("Restoring best checkpoint …")
        model.load_state_dict({k: v.to(config.DEVICE) for k, v in best_state.items()})

    print("\n─── Test Set Evaluation ─────────────────────────────────────────")
    sims, labels = evaluate_verification(
        model, test_loader, config.DEVICE, config.USE_LEXICAL_FEATURES,
    )

    eer_test, thr_test = compute_eer(sims, labels)
    auc_test           = compute_auc(sims, labels)

    # Accuracy at best threshold
    preds   = (sims >= best_threshold).astype(int)
    acc     = (preds == labels).mean()

    # Stats
    pos_sims = sims[labels == 1]
    neg_sims = sims[labels == 0]

    print(f"\n  Test EER           : {eer_test:.4f}")
    print(f"  Test AUC-ROC       : {auc_test:.4f}")
    print(f"  Accuracy @thr={best_threshold:.3f}: {acc:.4f}")
    print(f"\n  Same-author sims   : mean={pos_sims.mean():.4f}  std={pos_sims.std():.4f}")
    print(f"  Diff-author sims   : mean={neg_sims.mean():.4f}  std={neg_sims.std():.4f}")
    print(f"  Separation gap     : {pos_sims.mean() - neg_sims.mean():.4f}")

    # Per-threshold breakdown
    print(f"\n  {'Threshold':>10}  {'Accuracy':>8}  {'FAR':>6}  {'FRR':>6}")
    print("  " + "─" * 36)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (sims >= t).astype(int)
        a = (p == labels).mean()
        far = p[labels == 0].mean() if (labels == 0).any() else 0.0
        frr = 1.0 - p[labels == 1].mean() if (labels == 1).any() else 0.0
        print(f"  {t:>10.1f}  {a:>8.4f}  {far:>6.4f}  {frr:>6.4f}")

    # ── 7. Save checkpoint ────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    save_path = os.path.join("model", "siamese_verification.pt")
    torch.save({
        'model_state':     best_state or model.state_dict(),
        'vocab':           vocab,
        'author2idx':      author2idx,
        'lex_feature_dim': lex_dim,
        'threshold':       best_threshold,
        'eer':             best_eer,
        'config':          {k: v for k, v in config.__dict__.items()
                           if not k.startswith('_')},
    }, save_path)
    print(f"\n✓ Checkpoint saved → {save_path}")

    # ── 8. PCA Embedding Visualizations ────────────────────────────────────
    print("\n─── Generating PCA Embedding Visualizations ────────────────────")
    visualize_embeddings(
        model      = model,
        ds         = train_ds,
        config     = config,
        author2idx = author2idx,
        tag        = "trained",
        device     = config.DEVICE,
    )

    # ── 9. Demo: Verification function ────────────────────────────────────
    print("\n─── Verification Demo ──────────────────────────────────────────")
    print("Usage after loading checkpoint:")
    print("  model, vocab, lex_ext, threshold = load_verifier('model/siamese_verification.pt')")
    print("  same, confidence = verify(model, vocab, lex_ext, threshold, code_a, code_b)")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Inference API
# ─────────────────────────────────────────────────────────────────────────────

def load_verifier(
    path: str, device: str = "cpu",
) -> Tuple[CodeEmbeddingNet, CharVocabulary, Optional[LexicalFeatureExtractor], float]:
    """Load a trained verification model from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt['config']

    vocab = ckpt['vocab']
    lex_dim = ckpt['lex_feature_dim']
    lex_ext = LexicalFeatureExtractor() if lex_dim > 0 else None

    model = CodeEmbeddingNet(
        vocab_size      = len(vocab),
        embed_dim       = cfg.get('CHAR_EMBED_DIM', 64),
        hidden_dim      = cfg.get('LSTM_HIDDEN_DIM', 256),
        num_layers      = cfg.get('LSTM_NUM_LAYERS', 2),
        dropout         = 0.0,
        lex_feature_dim = lex_dim,
        projection_dim  = cfg.get('PROJECTION_DIM', 128),
    ).to(device)

    model.load_state_dict(
        {k: v.to(device) for k, v in ckpt['model_state'].items()}
    )
    model.eval()

    threshold = ckpt.get('threshold', 0.5)
    return model, vocab, lex_ext, threshold


@torch.no_grad()
def get_embedding(
    model:     CodeEmbeddingNet,
    vocab:     CharVocabulary,
    lex_ext:   Optional[LexicalFeatureExtractor],
    code:      str,
    max_len:   int  = 2000,
    device:    str  = "cpu",
) -> torch.Tensor:
    """Get the L2-normalised embedding for a single code snippet."""
    token_ids, length = vocab.encode(code, max_len)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths   = torch.tensor([max(length, 1)], dtype=torch.long, device=device)

    lex_feats = None
    if lex_ext is not None:
        feats     = lex_ext.extract(code)
        lex_feats = torch.tensor([feats], dtype=torch.float, device=device)

    return model(token_ids, lengths, lex_feats).squeeze(0)


def verify(
    model:     CodeEmbeddingNet,
    vocab:     CharVocabulary,
    lex_ext:   Optional[LexicalFeatureExtractor],
    threshold: float,
    code_a:    str,
    code_b:    str,
    max_len:   int = 2000,
    device:    str = "cpu",
) -> Tuple[bool, float]:
    """
    Verify whether two code snippets were written by the same author.

    Returns:
        (same_author, confidence)
        - same_author: True if predicted same author
        - confidence:  cosine similarity ∈ [-1, 1]
    """
    emb_a = get_embedding(model, vocab, lex_ext, code_a, max_len, device)
    emb_b = get_embedding(model, vocab, lex_ext, code_b, max_len, device)
    sim   = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
    return sim >= threshold, sim


if __name__ == '__main__':
    main()
