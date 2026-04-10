import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List

# Import components from multisiam
from multisiamese.multisiam import (
    MultiSiamConfig, 
    CharVocabulary, 
    MultiSiamEncoder, 
    LexicalFeatureExtractor
)

def get_embeddings_batched(model, codes: List[str], vocab, lex, device, batch_size=32):
    model.eval()
    all_embeddings = []
    
    for i in range(0, len(codes), batch_size):
        batch_codes = codes[i : i + batch_size]
        all_ids, all_lens, all_lex = [], [], []
        
        for c in batch_codes:
            ids, length = vocab.encode(c, 1000)
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_lens.append(length)
            if lex: all_lex.append(torch.tensor(lex.extract(c), dtype=torch.float))
        
        ids_t = torch.stack(all_ids).to(device)
        lens_t = torch.tensor(all_lens).to(device)
        lex_t = torch.stack(all_lex).to(device) if all_lex else None
        
        with torch.no_grad():
            emb = model(ids_t, lens_t, lex_t).cpu().numpy()
            all_embeddings.append(emb)
            
    return np.vstack(all_embeddings)

def main():
    cfg = MultiSiamConfig()
    device = cfg.DEVICE
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Load Data
    from multisiamese.siamese import load_raw_data
    df, author2idx = load_raw_data(cfg)
    
    # 2. Select Authors for Visualization
    author_list = df['label'].unique()[:30] # Visualize 30 authors
    samples = []
    for auth_id in author_list:
        sub = df[df['label'] == auth_id]
        samples.append(sub.sample(min(len(sub), 20))) # Use 20 snippets each to keep plot clean
    
    vis_df = pd.concat(samples).reset_index(drop=True)
    print(f"Data sampled: {len(vis_df)} rows for {len(author_list)} authors.")
    print(f"Columns in vis_df: {vis_df.columns.tolist()}")
    
    codes = vis_df[cfg.CODE_COLUMN].tolist()
    labels = vis_df['label'].tolist()
    author_names = [list(author2idx.keys())[list(author2idx.values()).index(l)] for l in labels]

    # 3. Initialize Models
    # Load Weights and Metadata
    checkpoint = torch.load("multisiam_author_model.pt", weights_only=False)
    vocab = checkpoint['vocab']
    
    # Adaptive Architecture: Check weight shape to detect if lex_feats were used
    proj_weight = checkpoint['model_state']['projection.0.weight']
    actual_fusion_dim = proj_weight.shape[1]
    base_lstm_dim = cfg.HIDDEN_DIM * 2
    detected_lex_dim = actual_fusion_dim - base_lstm_dim
    
    print(f"Detected fusion dim: {actual_fusion_dim} (LSTM: {base_lstm_dim}, Lex: {detected_lex_dim})")
    
    # Initialize Models with the DETECTED lex_dim
    random_model = MultiSiamEncoder(
        len(vocab), cfg.EMBED_DIM, cfg.HIDDEN_DIM, 
        cfg.NUM_LAYERS, cfg.DROPOUT, 
        detected_lex_dim, cfg.OUT_DIM
    ).to(device)
    
    trained_model = MultiSiamEncoder(
        len(vocab), cfg.EMBED_DIM, cfg.HIDDEN_DIM, 
        cfg.NUM_LAYERS, cfg.DROPOUT, 
        detected_lex_dim, cfg.OUT_DIM
    ).to(device)
    
    trained_model.load_state_dict(checkpoint['model_state'])
    
    # Setup Lexical Extractor if detected
    lex = LexicalFeatureExtractor() if detected_lex_dim > 0 else None

    # 4. Extract Embeddings (using Batched Inference to avoid OOM)
    print(f"Extracting embeddings for {len(codes)} snippets (Batch size: 32)...")
    embs_initial = get_embeddings_batched(random_model, codes, vocab, lex, device, batch_size=32)
    embs_trained = get_embeddings_batched(trained_model, codes, vocab, lex, device, batch_size=32)

    # 5. t-SNE Dimensionality Reduction
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=seed)
    
    # Combine to ensure same scale/projection
    combined = np.vstack([embs_initial, embs_trained])
    reduced = tsne.fit_transform(combined)
    
    reduced_initial = reduced[:len(embs_initial)]
    reduced_trained = reduced[len(embs_initial):]

    # 6. Plotting
    print("Generating graphs...")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Palette for 30 authors
    unique_authors = sorted(list(set(author_names)))
    palette = sns.color_palette("husl", len(unique_authors))

    # Plot Initial
    sns.scatterplot(
        x=reduced_initial[:, 0], y=reduced_initial[:, 1], 
        hue=author_names, ax=axes[0], palette=palette, alpha=0.6, s=30
    )
    axes[0].set_title("Embeddings BEFORE Training (30 Authors)", fontsize=18)
    axes[0].legend().remove()

    # Plot Trained
    sns.scatterplot(
        x=reduced_trained[:, 0], y=reduced_trained[:, 1], 
        hue=author_names, ax=axes[1], palette=palette, alpha=0.6, s=30
    )
    axes[1].set_title("Embeddings AFTER MultiSiam Training (30 Authors)", fontsize=18)
    # Put legend outside to the right
    axes[1].legend(title="Authors", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.tight_layout()
    plt.savefig("embedding_analysis_30.png", dpi=300, bbox_inches='tight')
    print("Visualization saved to embedding_analysis_30.png")

if __name__ == "__main__":
    main()
