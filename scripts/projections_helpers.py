import pandas as pd
import tempun
import random
import os
import numpy as np
import json
import importlib
from pathlib import Path
import requests
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import umap



def clean(x):
    return "" if pd.isna(x) else str(x)

def produce_hover(row):
    not_before = row.get("not_before")
    not_after = row.get("not_after")
    # Convert to int only if not NaN
    not_before = int(not_before) if pd.notna(not_before) else "?"
    not_after = int(not_after) if pd.notna(not_after) else "?"
    return "'{}' ({}, {}, [{}-{}])".format(
        clean(row.get("kwic_text")),
        clean(row.get("author")),
        clean(row.get("title")),
        not_before,
        not_after
    )


def get_random_date(row):
    random_date = tempun.model_date(row["not_before"], row["not_after"]),
    if isinstance(random_date, (list, tuple, np.ndarray)):
        random_date = random_date[0]
    return random_date

REDUCTIONS = {
    # --- UMAP variants ---
    "umap_10_0.05": lambda: umap.UMAP(n_neighbors=10,  min_dist=0.05,
                                      metric="cosine", random_state=42),
    "umap_50_0.5" : lambda: umap.UMAP(n_neighbors=50,  min_dist=0.5 ,
                                      metric="cosine", random_state=42),
    "umap_100_0.8": lambda: umap.UMAP(n_neighbors=100, min_dist=0.8,
                                      metric="cosine", random_state=42),
    # --- t-SNE variants ---
    "tsne_10": lambda: TSNE(n_components=2, perplexity=10,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    "tsne_30": lambda: TSNE(n_components=2, perplexity=30,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    "tsne_50": lambda: TSNE(n_components=2, perplexity=50,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    # --- PCA variants ---
    "pca_std"   : lambda: PCA(n_components=2, whiten=False , random_state=42),
    "pca_whiten": lambda: PCA(n_components=2, whiten=True  , random_state=42),
    # --- PCA→t-SNE speed trick ---
    "pca10_tsne": lambda: (
        lambda X: TSNE(n_components=2, perplexity=30,
                       init="pca", random_state=42)
                  .fit_transform(PCA(n_components=10, random_state=42)
                                 .fit_transform(X))
    )
}

def get_2d(emb, projection):
    reducer = projection()
    if hasattr(reducer, "fit_transform"):
            embedding_2d = reducer.fit_transform(emb)
    else:                      # callable expecting X directly (pca10_tsne)
            embedding_2d = reducer(emb)
    return embedding_2d[:, 0], embedding_2d[:, 1]


# =============================================================================
# Supervised Dimensionality Reduction (LDA)
# =============================================================================

class LDAProjection:
    """
    Wrapper for Linear Discriminant Analysis that can be used like the 
    unsupervised REDUCTIONS lambdas, but requires labels during fit.
    
    Usage:
        lda_proj = LDAProjection(n_components=2, random_state=42)
        X_2d = lda_proj.fit_transform(X_embeddings, y_labels)
        x, y = X_2d[:, 0], X_2d[:, 1]
        
        # Later, transform new data using the fitted model:
        X_new_2d = lda_proj.transform(X_new_embeddings)
    """
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit LDA on embeddings X and labels y."""
        self.model = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver='svd'  # SVD is more stable for high-dimensional data
        )
        self.model.fit(X, y)
        return self
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.model.transform(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.transform(X)
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Return absolute sum of weights across discriminants for each original dimension.
        Higher values indicate dimensions more important for class separation.
        """
        if self.model is None or not hasattr(self.model, 'scalings_'):
            raise ValueError("Model not fitted yet.")
        # scalings_: shape (n_features, n_components)
        return np.abs(self.model.scalings_).sum(axis=1)
    
    def get_top_dimensions(self, k: int = 10) -> np.ndarray:
        """Return indices of top-k most discriminative original dimensions."""
        importances = self.get_feature_importances()
        return np.argsort(importances)[-k:][::-1]


def compute_lda_projection(X: np.ndarray, y: np.ndarray, n_components: int = 2, random_state: int = 42):
    """
    Convenience function to compute LDA projection.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Embeddings (e.g., 768-dim PhilBerta vectors)
    y : np.ndarray of shape (n_samples,)
        Sense labels (integers)
    n_components : int
        Number of discriminant axes (max is n_classes - 1)
    random_state : int
        For reproducibility
        
    Returns
    -------
    X_2d : np.ndarray of shape (n_samples, n_components)
        Projected embeddings in LDA space
    lda_model : LDAProjection
        Fitted model (can be reused for new data)
    """
    lda = LDAProjection(n_components=n_components, random_state=random_state)
    X_2d = lda.fit_transform(X, y)
    return X_2d, lda


def evaluate_lda_vs_unsupervised(X: np.ndarray, y: np.ndarray, 
                                  unsup_methods: dict = None) -> dict:
    """
    Compare LDA against unsupervised methods (PCA, UMAP, etc.) using silhouette scores.
    
    Parameters
    ----------
    X : np.ndarray
        Embeddings
    y : np.ndarray
        Labels
    unsup_methods : dict
        Dict of {name: lambda} for unsupervised reducers (like REDUCTIONS)
        
    Returns
    -------
    results : dict
        {method_name: silhouette_score}
    """
    from sklearn.metrics import silhouette_score
    
    if unsup_methods is None:
        unsup_methods = {}
    
    results = {}
    
    # LDA
    X_lda, _ = compute_lda_projection(X, y, n_components=2)
    results['LDA'] = silhouette_score(X_lda, y)
    
    # Unsupervised methods
    for name, reducer_fn in unsup_methods.items():
        try:
            X_2d = reducer_fn().fit_transform(X)
            results[name] = silhouette_score(X_2d, y)
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    return results


# =============================================================================
# Label-Based Projection Quality Metrics
# =============================================================================

def silhouette_from_labels(X_2d: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette coefficient using LLM labels as ground truth clusters.
    Higher = better separation in 2D space.

    For each point i:
      a = mean distance to same-label points
      b = mean distance to nearest other-label cluster
      s(i) = (b - a) / max(a, b)

    Returns mean over all points. Range: [-1, 1].
    """
    return silhouette_score(X_2d, labels)


def davies_bouldin_from_labels(X_2d: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin index using LLM labels.
    Lower = better separation (small intra-cluster, large inter-cluster).
    """
    return davies_bouldin_score(X_2d, labels)


def calinski_harabasz_from_labels(X_2d: np.ndarray, labels: np.ndarray) -> float:
    """
    Calinski-Harabasz variance ratio using LLM labels.
    Higher = better separation.
    """
    return calinski_harabasz_score(X_2d, labels)


def intra_inter_label_ratio(X_2d: np.ndarray, labels: np.ndarray) -> float:
    """
    Your proposed metric: average intra-label distance / average total distance.

    For each point i:
      - avg_same[i] = mean Euclidean distance to all points with same label
      - avg_all[i]  = mean Euclidean distance to all points

    Score = mean_i (avg_same[i] / avg_all[i])

    Lower = better (same-label points are closer than average).
    """
    n = len(labels)
    ratios = []

    for i in range(n):
        dists = np.linalg.norm(X_2d - X_2d[i], axis=1)
        same_mask = labels == labels[i]

        # Avoid self-distance
        avg_same = dists[same_mask].sum() / (same_mask.sum() - 1)
        avg_all = dists.sum() / (n - 1)

        if avg_all > 0:
            ratios.append(avg_same / avg_all)

    return np.mean(ratios)


def evaluate_projection_2d(X_2d: np.ndarray, labels: np.ndarray) -> dict:
    """
    Comprehensive evaluation of a 2D projection w.r.t. label separation.

    Returns dict with:
      - silhouette: higher is better
      - davies_bouldin: lower is better
      - calinski_harabasz: higher is better
      - intra_inter_ratio: lower is better (your custom metric)
    """
    X_2d_arr = np.column_stack(X_2d) if not isinstance(X_2d, np.ndarray) else X_2d

    return {
        "silhouette": silhouette_from_labels(X_2d_arr, labels),
        "davies_bouldin": davies_bouldin_from_labels(X_2d_arr, labels),
        "calinski_harabasz": calinski_harabasz_from_labels(X_2d_arr, labels),
        "intra_inter_ratio": intra_inter_label_ratio(X_2d_arr, labels),
    }


# =============================================================================
# Reusable Plotting Functions
# =============================================================================

def create_label_palette(n_classes: int, seed: int = 42) -> dict[int, str]:
    """
    Generate a visually distinct color palette for n_classes.

    Parameters
    ----------
    n_classes : int
        Number of classes
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Mapping {class_id: hex_color}
    """
    rng = np.random.default_rng(seed)
    colors = [f"#{rng.integers(0, 256):02x}{rng.integers(0, 256):02x}{rng.integers(0, 256):02x}"
              for _ in range(n_classes)]
    return dict(enumerate(colors))


def plot_single_2d_projection(X_2d: tuple[np.ndarray, np.ndarray],
                               labels: np.ndarray,
                               label_names: dict[int, str],
                               palette: dict[int, str] | None = None,
                               title: str = "",
                               figsize: tuple[int, int] = (10, 10)) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a scatter plot of 2D projection colored by labels.

    Parameters
    ----------
    X_2d : tuple of arrays (x, y)
        2D coordinates
    labels : array of shape (n,)
        Integer label IDs
    label_names : dict
        Label ID to display name mapping
    palette : dict, optional
        Custom color mapping {label_id: hex_color}
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    x, y = X_2d
    n_classes = max(labels) + 1

    if palette is None:
        palette = create_label_palette(n_classes)

    colors = [palette.get(l, "#cccccc") for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=colors, s=8, alpha=0.6)

    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")

    # Legend
    handles = [
        Line2D([0], [0], marker='o', linestyle='', color=palette[lid], label=name, markersize=8)
        for lid, name in sorted(label_names.items())
    ]
    fig.legend(handles=handles, ncol=min(5, len(handles)),
               bbox_to_anchor=(0.5, -0.03), loc="upper center")
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return fig, ax


def plot_projection_grid(emb_array: np.ndarray,
                         labels: np.ndarray,
                         label_names: dict[int, str],
                         REDUCTIONS: dict,
                         palette: dict[int, str] | None = None,
                         figsize: tuple[int, int] = (13, 13)) -> plt.Figure:
    """
    Create 3x3 grid of projections showing all reduction methods.

    Parameters
    ----------
    emb_array : np.ndarray of shape (n, d)
        Original embeddings
    labels : array of shape (n,)
        Integer label IDs
    label_names : dict
        Label ID to display name mapping
    REDUCTIONS : dict
        Projection configurations (like in notebook)
    palette : dict, optional
        Color mapping {label_id: hex_color}
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib figure
    """
    n_classes = max(labels) + 1
    if palette is None:
        palette = create_label_palette(n_classes)

    PROJECTION_GROUPS = {
        "UMAP":  ["umap_10_0.05", "umap_50_0.5", "umap_100_0.8"],
        "t-SNE": ["tsne_10",      "tsne_30",     "tsne_50"],
        "PCA":   ["pca_std",      "pca_whiten",  "pca10_tsne"],
    }

    colors = [palette.get(l, "#cccccc") for l in labels]

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("2-D Projections", fontsize=16)

    for row_idx, (tech, keys) in enumerate(PROJECTION_GROUPS.items()):
        for col_idx, key in enumerate(keys):
            ax = axes[row_idx, col_idx]
            x, y = get_2d(emb_array, REDUCTIONS[key])
            ax.scatter(x, y, c=colors, s=8, alpha=0.6)
            ax.set_title(f"{tech} — {key}", fontsize=9)
            ax.axis("off")

    # Legend
    handles = [
        Line2D([0], [0], marker='o', linestyle='', color=palette[lid], label=name, markersize=8)
        for lid, name in sorted(label_names.items())
    ]
    legend = fig.legend(handles=handles, ncol=min(5, len(handles)),
                        bbox_to_anchor=(0.5, -0.02), loc="upper center")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def create_3d_temporal_projection(emb_array: np.ndarray,
                                   dates: np.ndarray,
                                   labels: np.ndarray,
                                   label_names: dict[int, str],
                                   REDUCTIONS: dict,
                                   proj_name: str = "pca10_tsne",
                                   palette: dict[int, str] | None = None,
                                   figsize: tuple[int, int] = (12, 9)) -> tuple[plt.Figure, plt.Axes]:
    """
    Create 3D scatter plot with time on X-axis, 2D projection on Y-Z plane.

    Parameters
    ----------
    emb_array : np.ndarray of shape (n, d)
        Original embeddings
    dates : array-like of shape (n,)
        Random dates from tempun.model_date
    labels : array-like of shape (n,)
        Integer label IDs
    label_names : dict
        Label ID to display name mapping
    REDUCTIONS : dict
        Projection configurations
    proj_name : str
        Which projection to use (key from REDUCTIONS)
    palette : dict, optional
        Color mapping {label_id: hex_color}
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib 3D figure and axes
    """
    n_classes = max(labels) + 1
    if palette is None:
        palette = create_label_palette(n_classes)

    # Compute 2D projection
    y, z = get_2d(emb_array, REDUCTIONS[proj_name])
    x = dates

    # Handle NaN dates
    valid_mask = ~np.isnan(x)
    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
    labels_valid = labels[valid_mask]
    colors = [palette.get(l, "#cccccc") for l in labels_valid]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_valid, y_valid, z_valid, c=colors, s=15, alpha=0.7)

    ax.set_title(f"3D Temporal Projection ({proj_name})", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Projection Dimension 1")
    ax.set_zlabel("Projection Dimension 2")

    if len(x_valid) > 0:
        ax.set_xlim(min(x_valid) - 20, max(x_valid) + 20)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=str(name),
               markerfacecolor=palette[lid], markersize=8)
        for lid, name in sorted(label_names.items())
    ]
    ax.legend(handles=legend_elements, title="Sense Classes",
              bbox_to_anchor=(0.85, 1), loc='upper left')

    plt.tight_layout()
    return fig, ax