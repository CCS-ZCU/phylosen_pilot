# classification_helpers.py — Reusable WSD classification utilities for phylosen_pilot
# Extracted and generalized from labyrinth/scripts/wsd_io.py and helpers.py

from __future__ import annotations

import re, os, json, random, math
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from torch import amp as torch_amp
from itertools import zip_longest


# =============================================================================
# Tokenization & Span Extraction (Generalized)
# =============================================================================

MAX_LEN = 256


def _ws(s: str) -> List[str]:
    """Split string on whitespace."""
    return s.strip().split() if s and pd.notna(s) else []


def _tokens_and_target_idx(left: str, target: str, right: str) -> Tuple[List[str], int]:
    """Build token list [left..., target, right...] and return target index."""
    lt = _ws(left)
    rt = _ws(right)
    return lt + [target] + rt, len(lt)


def _pad_trunc_enc(enc: dict, max_len: int, pad_id: int = 0) -> dict:
    """Pad or truncate encoding to max_len."""
    ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)

    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    if attn is None:
        attn = torch.ones_like(ids, dtype=torch.long)
    elif not torch.is_tensor(attn):
        attn = torch.tensor(attn, dtype=torch.long).unsqueeze(0)

    L = ids.shape[1]
    if L < max_len:
        pad_len = max_len - L
        ids = torch.cat([ids, torch.full((1, pad_len), pad_id, dtype=ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.zeros((1, pad_len), dtype=attn.dtype)], dim=1)
    elif L > max_len:
        ids = ids[:, :max_len]
        attn = attn[:, :max_len]

    enc["input_ids"] = ids
    enc["attention_mask"] = attn
    return enc


def span_xlmr(tokenizer, tokens: List[str], target_idx: int, max_length: int = MAX_LEN) -> List[int]:
    """
    Get subword span for XLM-R pretokenized input using word_ids alignment.
    Returns list of subword indices corresponding to the target token.
    """
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=max_length)
    word_ids = enc.word_ids(0)
    return [] if word_ids is None else [i for i, wid in enumerate(word_ids) if wid == target_idx]


def span_generic(tokenizer, tokens: List[str], target_idx: int, max_length: int = MAX_LEN) -> List[int]:
    """
    Get subword span for generic tokenizers (e.g., PhilBerta, Qwen).
    Uses string-based encoding with special token offset detection.
    """
    # Get subword lengths per whitespace token
    wp_lens: List[int] = []
    for tok in tokens:
        try:
            ids = tokenizer.encode(tok, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(tok)
        wp_lens.append(len(ids))

    sent_str = " ".join(tokens)
    enc = tokenizer(sent_str, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=max_length)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    enc = _pad_trunc_enc(enc, max_length, pad_id=pad_id)
    seq_len = int(enc["attention_mask"][0].sum().item())

    # Detect special token offset
    try:
        core_ids = tokenizer.encode(sent_str, add_special_tokens=False)
    except TypeError:
        core_ids = tokenizer.encode(sent_str)

    left_specials = 1
    try:
        built = tokenizer.build_inputs_with_special_tokens(core_ids)
        start = _first_occurrence(built, core_ids)
        if start is not None:
            left_specials = start
    except Exception:
        try:
            with_spec = tokenizer.encode(sent_str, add_special_tokens=True)
            start = _first_occurrence(with_spec, core_ids)
            if start is not None:
                left_specials = start
        except Exception:
            pass

    start_in_core = sum(wp_lens[:target_idx])
    k = wp_lens[target_idx] if target_idx < len(wp_lens) else 0
    if k <= 0:
        return []
    start = left_specials + start_in_core
    end = start + k - 1
    return [i for i in range(start, end + 1) if 0 <= i < seq_len]


def _first_occurrence(hay: List[int], needle: List[int]) -> Optional[int]:
    """Find first occurrence of needle in hay."""
    if not needle:
        return None
    n, m = len(hay), len(needle)
    for i in range(0, n - m + 1):
        if hay[i:i+m] == needle:
            return i
    return None


# =============================================================================
# Embedding Extraction (Flexible for Different Models/Layers)
# =============================================================================

@dataclass
class EmbeddingResult:
    """Container for embedding extraction results."""
    embedding: np.ndarray
    span: List[int]
    sentence_str: str
    subwords: List[str]


def extract_contextual_embedding(
    instance: pd.Series,
    tokenizer,
    model,
    device: str = "cuda",
    layer_idx: int = -3,
    piece_pooling: str = "mean",
    context_lemmatized: bool = True,
    target_lemmatized: bool = True,
    context_pos_filtered: bool = False,
    pos_keep: Set[str] = None,
    model_kind: str = "xlmr",  # "xlmr" | "labert" | "generic"
    max_length: int = MAX_LEN,
) -> EmbeddingResult:
    """
    Extract contextual embedding for a single instance.

    Parameters:
    -----------
    instance : pd.Series
        Must contain: left_context, right_context, target_word, target_lemma (optional)
                     tokens (optional), lemma (optional for target)
    tokenizer : HuggingFace-style tokenizer
    model : HuggingFace model with output_hidden_states=True
    device : "cuda" or "cpu"
    layer_idx : Layer index (negative = from end, e.g., -1 = last)
    piece_pooling : "mean" or "sum"
    context_lemmatized : Use lemmas for context tokens
    target_lemmatized : Use lemma for target token
    context_pos_filtered : Filter context by POS tags
    pos_keep : Set of POS tags to keep (default: {"NOUN", "PROPN", "ADJ", "VERB"})
    model_kind : Tokenization strategy ("xlmr", "labert", "generic")
    max_length : Max sequence length

    Returns:
    --------
    EmbeddingResult with embedding array, span, sentence string, and subwords
    """
    if pos_keep is None:
        pos_keep = {"NOUN", "PROPN", "ADJ", "VERB"}

    # Build token list from instance
    if context_lemmatized and hasattr(instance, 'tokens') and instance.tokens:
        # Use token list from instance
        tokens_list = json.loads(instance.tokens) if isinstance(instance.tokens, str) else instance.tokens
        if context_pos_filtered:
            tokens_list = [t for t in tokens_list if t.get('pos', '') in pos_keep]

        # Find target token
        target_lemma = instance.get('target_lemma', instance.get('lemma', ''))
        target_token = None
        for t in tokens_list:
            if t.get('lemma', '').lower() == target_lemma.lower():
                target_token = t['lemma'] if target_lemmatized else t['token_text']
                break

        if target_token is None:
            target_token = instance.target_lemmatized if target_lemmatized else instance.target_word

        # Build left/right context
        target_idx = None
        left_tokens = []
        for t in tokens_list:
            if t.get('lemma', '').lower() == target_lemma.lower():
                target_idx = len(left_tokens)
                break
            left_tokens.append(t['lemma'] if context_lemmatized else t['token_text'])

        if target_idx is None:
            target_idx = len(left_tokens)

        right_tokens = [t['lemma' if context_lemmatized else 'token_text']
                       for t in tokens_list[target_idx+1:]]
        left_str = " ".join(left_tokens)
        right_str = " ".join(right_tokens)
    else:
        # Use string-based context
        left_str = instance.get('left_context', '')
        right_str = instance.get('right_context', '')
        target_token = instance.get('target_lemma', '') if target_lemmatized else instance.get('target_word', '')

    # Build token list and get target index
    tokens, target_idx = _tokens_and_target_idx(left_str, target_token, right_str)

    # Get span based on model kind
    if model_kind == "xlmr":
        span = span_xlmr(tokenizer, tokens, target_idx, max_length)
    else:
        span = span_generic(tokenizer, tokens, target_idx, max_length)

    # Encode and get hidden states
    enc = tokenizer(tokens, is_split_into_words=(model_kind == "xlmr"),
                   return_tensors="pt", padding="max_length", truncation=True,
                   max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outs = model(**enc)
        hidden = outs.hidden_states  # tuple of layers

        # Select layer
        idx = layer_idx if layer_idx >= 0 else len(hidden) + layer_idx
        layer_hidden = hidden[idx].squeeze(0)  # [L, D]

    # Pool over span
    if span:
        span_tensor = layer_hidden[span]  # [K, D]
        if piece_pooling == "sum":
            embedding = span_tensor.sum(dim=0).cpu().numpy()
        else:
            embedding = span_tensor.mean(dim=0).cpu().numpy()
    else:
        # Fallback to CLS
        embedding = layer_hidden[0].cpu().numpy()

    # Get subwords for debugging
    try:
        subwords = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    except:
        subwords = []

    sentence_str = " ".join(tokens)

    return EmbeddingResult(
        embedding=embedding.astype(np.float32),
        span=span,
        sentence_str=sentence_str,
        subwords=subwords
    )


def batch_extract_embeddings(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str = "cuda",
    layer_idx: int = -3,
    piece_pooling: str = "mean",
    context_lemmatized: bool = True,
    target_lemmatized: bool = True,
    context_pos_filtered: bool = False,
    pos_keep: Set[str] = None,
    model_kind: str = "xlmr",
    max_length: int = MAX_LEN,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract embeddings for all rows in a DataFrame.

    Returns:
    --------
    np.ndarray of shape (n_samples, hidden_size)
    """
    embeddings = []
    for _, row in df.iterrows():
        result = extract_contextual_embedding(
            row, tokenizer, model, device=device, layer_idx=layer_idx,
            piece_pooling=piece_pooling, context_lemmatized=context_lemmatized,
            target_lemmatized=target_lemmatized, context_pos_filtered=context_pos_filtered,
            pos_keep=pos_keep, model_kind=model_kind, max_length=max_length
        )
        embeddings.append(result.embedding)

    return np.stack(embeddings, axis=0)


# =============================================================================
# MLP Classifier (Simplified from labyrinth)
# =============================================================================

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def create_mlp_classifier(hidden_sizes: Tuple[int, ...] = (512, 128),
                          alpha: float = 1e-4,
                          max_iter: int = 500,
                          random_state: int = 42) -> MLPClassifier:
    """
    Create an MLP classifier with specified architecture.

    Parameters:
    -----------
    hidden_sizes : Tuple of hidden layer sizes
    alpha : L2 regularization strength
    max_iter : Maximum iterations
    random_state : Random seed

    Returns:
    --------
    Configured MLPClassifier
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state
    )


def evaluate_classifier(X: np.ndarray, y: np.ndarray, classifier,
                        cv: int = 5, return_model: bool = False,
                        scale: bool = True) -> Dict[str, Any]:
    """
    Evaluate classifier with cross-validation.

    Parameters:
    -----------
    X : Feature matrix (n_samples, n_features)
    y : Labels (n_samples,)
    classifier : Scikit-learn classifier
    cv : Number of CV folds
    return_model : If True, also train and return final model
    scale : Whether to standardize features

    Returns:
    --------
    Dict with 'predictions', 'accuracy', 'f1_macro', and optionally 'model'
    """
    if scale:
        pipeline = make_pipeline(StandardScaler(), clone(classifier))
    else:
        pipeline = clone(classifier)

    # Cross-validated predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    result = {
        'predictions': y_pred,
        'accuracy': acc,
        'f1_macro': f1
    }

    if return_model:
        pipeline.fit(X, y)
        result['model'] = pipeline

    return result


# =============================================================================
# Data Preparation Helpers
# =============================================================================

def prepare_classification_data(df: pd.DataFrame,
                                emb_col: str,
                                label_col: str = "classification_single_label") -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X, y for classification from a DataFrame with embedded columns.

    Parameters:
    -----------
    df : DataFrame with embedding and label columns
    emb_col : Column name containing embedding arrays
    label_col : Column name containing class labels

    Returns:
    --------
    X : Feature matrix (n_samples, embedding_dim)
    y : Label array (n_samples,)
    """
    X = np.stack(df[emb_col].values)

    # Encode labels if they're strings
    if df[label_col].dtype == object:
        unique_labels = df[label_col].unique()
        label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.array([label2idx[lbl] for lbl in df[label_col]])
    else:
        y = df[label_col].values

    return X, y


def merge_multiple_embeddings(df: pd.DataFrame,
                              emb_cols: List[str],
                              method: str = "mean") -> np.ndarray:
    """
    Merge multiple embedding columns into one.

    Parameters:
    -----------
    df : DataFrame with embedding columns
    emb_cols : List of column names containing embeddings
    method : "mean", "concat", or "weighted"

    Returns:
    --------
    Merged embedding matrix (n_samples, new_dim)
    """
    mats = [np.stack(df[col].values) for col in emb_cols]
    n_rows = [m.shape[0] for m in mats]

    if len(set(n_rows)) != 1:
        raise ValueError(f"Inconsistent row counts across embeddings: {n_rows}")

    if method == "concat":
        return np.concatenate(mats, axis=1)
    elif method == "mean":
        M = np.stack(mats, axis=0)
        return M.mean(axis=0)
    elif method == "weighted":
        raise NotImplementedError("Weighted merging requires explicit weights parameter")
    else:
        raise ValueError(f"Unknown method: {method}")
