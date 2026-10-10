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
from matplotlib.patches import Patch
import umap
from typing import Dict, Any, Optional, Tuple



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


def build_embedding_matrix(df, spec, method="concat"):
    """Assemble an embedding matrix from one or several array columns.

    Keeps the projection helpers agnostic to the encoder choice: the best WSD
    classifiers concatenate two encoders, so ``spec`` may name several columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing the embedding columns.
    spec : str or sequence of str
        A single column name, or several to be combined.
    method : {"concat", "mean"}
        How to combine several columns. ``"concat"`` (default) horizontally
        stacks them (result dim = sum of parts); ``"mean"`` averages them
        (requires equal dimensionality). Ignored for a single column.

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
    """
    cols = [spec] if isinstance(spec, str) else list(spec)
    if not cols:
        raise ValueError("Embedding spec is empty.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing embedding column(s): {missing}")

    mats = [np.stack(df[c].values) for c in cols]
    if len(mats) == 1:
        return mats[0]
    if method == "concat":
        return np.concatenate(mats, axis=1)
    if method == "mean":
        if len({m.shape[1] for m in mats}) != 1:
            raise ValueError("method='mean' requires equal-dimensional embeddings.")
        return np.mean(mats, axis=0)
    raise ValueError(f"Unknown method: {method!r} (use 'concat' or 'mean').")


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
                               figsize: tuple[int, int] = (10, 10),
                               ax: plt.Axes | None = None,
                               show_legend: bool = True) -> tuple[plt.Figure, plt.Axes]:
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
        Figure size (only used when ``ax`` is None)
    ax : matplotlib Axes, optional
        Existing axes to draw into. When None, a new standalone figure is
        created (original behaviour). When provided, ``figsize`` is ignored.
    show_legend : bool
        Whether to draw the figure-level legend below the axes. Set to False
        when composing this panel into a shared figure with a single legend.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    x, y = X_2d
    n_classes = max(labels) + 1

    if palette is None:
        palette = create_label_palette(n_classes)

    colors = [palette.get(l, "#cccccc") for l in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.scatter(x, y, c=colors, s=8, alpha=0.6)

    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")

    if show_legend:
        # Legend
        handles = [
            Line2D([0], [0], marker='o', linestyle='', color=palette[lid], label=name, markersize=8)
            for lid, name in sorted(label_names.items())
        ]
        fig.legend(handles=handles, ncol=min(5, len(handles)),
                   bbox_to_anchor=(0.5, -0.03), loc="upper center")

    if ax is None:
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


# =============================================================================
# Century-Based Sense Trajectories (3D Visualization)
# =============================================================================

def compute_century_trajectories_data(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    lda_model,
    date_column: str = 'random_date',
    min_samples: int = 5,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute trajectory data for all senses across centuries.
    
    Parameters
    ----------
    df : DataFrame with date_column and label column
    X : Embedding array aligned with df (n_samples, n_features)
    y : Label array aligned with df (n_samples,)
    lda_model : Fitted LDAProjection model
    date_column : Column name containing year values
    min_samples : Minimum samples per century to compute valid centroid
    
    Returns
    -------
    dict {sense_label: {
        'centuries': [101, 201, 301, ...],      # sorted century starts  
        'x': [x1, x2, x3, ...],                  # LD1 positions
        'y': [y1, y2, y3, ...],                  # LD2 positions
        'z': [z1, z2, z3, ...],                  # century values (Z axis)
        'marker_sizes': [s1, s2, s3, ...],       # log-scaled sizes
        'sample_counts': [n1, n2, n3, ...],      # raw counts per century
        'is_valid': [True, False, ...]           # True if >= min_samples
    }}
    
    Notes
    -----
    - Century calculation: (year // 100) * 100 + 1 (e.g., year 150 → 101)
    - Uses pre-fitted LDA model to project centroids
    - Only includes centuries with >= min_samples attestations
    """
    
    def year_to_century_midpoint(year):
        """
        Convert year to century midpoint.
        
        BCE centuries: 8th c. BCE (-800 to -701) → -750
        CE centuries:  2nd c. CE (101 to 200) → 150
        
        Year 0 is omitted (returns None).
        """
        if pd.isna(year):
            return None
        
        year = int(year)
        if year == 0:
            return None
        
        if year < 0:  # BCE
            # e.g., -750 → 8th c. BCE (-800 to -701) → midpoint -750
            # e.g., -700 → 7th c. BCE (-700 to -601) → midpoint -650
            century_num = (-year + 99) // 100
            midpoint = -(century_num * 100 - 50)
        else:  # CE
            # e.g., 150 → 2nd c. CE (101 to 200) → midpoint 150
            # e.g., 50 → 1st c. CE (1 to 100) → midpoint 50
            century_num = (year - 1) // 100 + 1
            midpoint = (century_num - 1) * 100 + 50
        
        return midpoint
    
    # Ensure X, y, and df are aligned by using min length
    n_min = min(len(df), len(X), len(y))
    
    # Build aligned arrays without index confusion
    centuries_list = []
    valid_X = []
    valid_y = []
    
    for idx in range(n_min):
        year = df[date_column].iloc[idx]
        midpoint = year_to_century_midpoint(year)
        if midpoint is not None:
            centuries_list.append(midpoint)
            valid_X.append(X[idx])
            valid_y.append(y[idx])
    
    if len(valid_X) == 0:
        return {}
    
    centuries_arr = np.array(centuries_list)
    X_valid = np.array(valid_X)
    y_valid = np.array(valid_y)
    
    trajectories = {}
    
    for label in np.unique(y_valid):
        label_mask = y_valid == label
        label_centuries = centuries_arr[label_mask]
        label_X = X_valid[label_mask]
        
        centuries = []
        x_vals = []
        y_vals = []
        z_vals = []
        marker_sizes = []
        sample_counts = []
        is_valid = []
        
        for midpoint in np.sort(np.unique(label_centuries)):
            century_mask = label_centuries == midpoint
            count = np.sum(century_mask)
            X_century = label_X[century_mask]
            
            centroid = X_century.mean(axis=0)
            centroid_2d = lda_model.transform(centroid.reshape(1, -1))[0]
            
            centuries.append(int(midpoint))
            x_vals.append(float(centroid_2d[0]))
            y_vals.append(float(centroid_2d[1]))
            z_vals.append(int(midpoint))  # Z-axis uses midpoint as position
            sample_counts.append(count)
            is_valid.append(count >= min_samples)
            
            # Log10 scaling for marker size
            size = 50 + 100 * np.log10(max(count, 1))
            marker_sizes.append(float(size))
        
        trajectories[int(label)] = {
            'centuries': centuries,
            'x': x_vals,
            'y': y_vals,
            'z': z_vals,
            'marker_sizes': marker_sizes,
            'sample_counts': sample_counts,
            'is_valid': is_valid,
        }
    
    return trajectories


def plot_3d_century_trajectories(
    trajectory_data: Dict[int, Dict[str, Any]],
    label_names: Optional[Dict[int, str]] = None,
    palette: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    interpolate_gaps: bool = True,
    max_gap_centuries: int = 2,
    title: str = "Sense Evolution Across Centuries",
    ax=None,
    show_legend: bool = True,
    elev: float = 20,
    azim: float = 45,
    marker: str = 'o',
    marker_size: Optional[float] = None,
) -> plt.Figure:
    """
    Render 3D trajectory plot from pre-computed century trajectory data.
    
    Parameters
    ----------
    trajectory_data : Output from compute_century_trajectories_data
    label_names : {sense_label: display_name} mapping (e.g., from llmprompt_passio.LABEL_MAP)
    palette : {sense_label: color} mapping
    figsize : Figure size (only used when ``ax`` is None)
    interpolate_gaps : Whether to connect gaps with gray lines
    max_gap_centuries : Max consecutive missing centuries to interpolate
    title : Plot title
    ax : matplotlib 3D Axes, optional
        Existing 3D axes to draw into. When None, a new standalone figure is
        created (original behaviour). When provided, ``figsize`` is ignored.
    show_legend : bool
        Whether to draw the axes legend. Set to False when composing this
        panel into a shared figure with a single legend.
    elev, azim : float
        Elevation and azimuth view angles passed to ``ax.view_init``.
    marker : Matplotlib marker shape, e.g. 'o' for passio or 'x' for pathos.
    marker_size : Fixed scatter area; None retains count-dependent sizes.
    
    Returns
    -------
    fig : matplotlib Figure with 3D trajectory plot
    
    Notes
    -----
    - X,Y = LDA space, Z = time (century midpoints: -750, -650, ..., 50, 150, ...)
    - Marker size encodes sample count (log-scaled)
    - Interpolated segments shown as gray dashed lines
    - All senses shown in single plot with legend
    """
    
    if palette is None:
        n_classes = len(trajectory_data)
        palette = create_label_palette(n_classes)
    
    if label_names is None:
        label_names = {label: f"Sense {label}" for label in trajectory_data.keys()}
    
    if ax is None:
        ax_owned = True
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax_owned = False
        fig = ax.figure

    previous_zlim = ax.get_zlim() if not ax_owned and ax.has_data() else None
    
    # Track all data for legend
    legend_handles = []
    
    # Determine Z-axis range for proper scaling
    all_z = []
    for data in trajectory_data.values():
        all_z.extend(data['z'])
    
    for label, data in sorted(trajectory_data.items()):
        centuries = data['centuries']
        x = data['x']
        y = data['y']
        z = data['z']
        sizes = data['marker_sizes']
        is_valid = data['is_valid']
        
        if len(centuries) == 0:
            continue
        
        color = palette.get(label, '#cccccc')
        label_display = label_names.get(label, f"Sense {label}")
        
        # Plot the trajectory line
        if interpolate_gaps:
            # Find gaps and interpolate
            valid_indices = np.where(is_valid)[0]
            
            if len(valid_indices) > 1:
                # Draw segments between valid points
                for i in range(len(valid_indices) - 1):
                    idx1, idx2 = valid_indices[i], valid_indices[i + 1]
                    
                    # Check gap size in terms of century steps
                    z_gap = abs(z[idx2] - z[idx1])
                    # Each century step is 100 years, so gap_centuries = z_gap / 100
                    gap_centuries = z_gap // 100
                    
                    if gap_centuries <= max_gap_centuries + 1:
                        # Connect these points with solid line
                        ax.plot(
                            [x[idx1], x[idx2]],
                            [y[idx1], y[idx2]],
                            [z[idx1], z[idx2]],
                            color=color,
                            alpha=1.0,
                            linewidth=1.5
                        )
                    elif gap_centuries <= max_gap_centuries * 2:
                        # Interpolate with gray dashed line
                        ax.plot(
                            [x[idx1], x[idx2]],
                            [y[idx1], y[idx2]],
                            [z[idx1], z[idx2]],
                            color='gray',
                            alpha=0.5,
                            linewidth=1.0,
                            linestyle='--'
                        )
        else:
            # Just connect adjacent valid points
            valid_indices = np.where(is_valid)[0]
            for i in range(len(valid_indices) - 1):
                idx1, idx2 = valid_indices[i], valid_indices[i + 1]
                ax.plot(
                    [x[idx1], x[idx2]],
                    [y[idx1], y[idx2]],
                    [z[idx1], z[idx2]],
                    color=color,
                    alpha=1.0,
                    linewidth=1.5
                )
        
        # Plot markers for each valid point
        valid_x = [x[i] for i, v in enumerate(is_valid) if v]
        valid_y = [y[i] for i, v in enumerate(is_valid) if v]
        valid_z = [z[i] for i, v in enumerate(is_valid) if v]
        valid_sizes = [sizes[i] if marker_size is None else marker_size
                       for i, v in enumerate(is_valid) if v]
        
        if valid_x:
            scatter = ax.scatter(
                valid_x, valid_y, valid_z,
                c=[color] * len(valid_x),
                s=valid_sizes,
                marker=marker,
                alpha=0.8,
                **({'linewidths': 1.5} if marker in ('x', '+') else
                   {'edgecolors': 'white', 'linewidths': 0.5})
            )
            
            # Create legend handle (only once per label)
            if not any(h.get_label() == label_display for h in legend_handles):
                legend_handles.append(
                    Line2D([0], [0], marker=marker, color=color,
                          linestyle='',
                          label=label_display,
                          markerfacecolor=color, markersize=8)
                )
    
    # Set labels and title
    ax.set_xlabel('LD1 (Discriminant Axis 1)', fontsize=11)
    ax.set_ylabel('LD2 (Discriminant Axis 2)', fontsize=11)
    ax.set_zlabel('Century (midpoint)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set Z-axis limits with some padding
    if all_z:
        z_min, z_max = min(all_z), max(all_z)
        z_padding = 50  # Half a century
        z_limits = (z_min - z_padding, z_max + z_padding)
        if previous_zlim is not None:
            z_limits = (min(z_limits[0], previous_zlim[0]),
                        max(z_limits[1], previous_zlim[1]))
        ax.set_zlim(*z_limits)
    
    # Legend
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
    
    # Adjust view angle for better visualization
    ax.view_init(elev=elev, azim=azim)
    
    if ax_owned:
        plt.tight_layout()
    return fig


# =============================================================================
# 3D LDA + Temporal Scatter Plot
# =============================================================================

def plot_3d_lda_temporal(X_lda: np.ndarray,
                         dates: np.ndarray,
                         labels: np.ndarray,
                         label_names: Optional[Dict[int, str]] = None,
                         palette: Optional[Dict[int, str]] = None,
                         title: str = "",
                         figsize: Tuple[int, int] = (12, 9),
                         ax=None,
                         show_legend: bool = True,
                         elev: float = 25,
                         azim: float = 45,
                         xlabel: str = 'LD1 (Discriminant Axis 1)',
                         ylabel: str = 'LD2 (Discriminant Axis 2)',
                         zlabel: str = 'Year (random_date)') -> Tuple[plt.Figure, plt.Axes]:
    """
    Create 3D scatter plot with a 2D projection (X,Y) and time (Z-axis).
    
    Despite the name, this works for any 2D projection: pass LDA coordinates for
    a supervised view, or PCA+t-SNE / UMAP coordinates for an unsupervised one.
    
    Parameters
    ----------
    X_lda : np.ndarray of shape (n, 2)
        2D projection coordinates (e.g. LD1, LD2 or PCA+t-SNE dims)
    dates : array-like of shape (n,)
        Random dates from tempun.model_date
    labels : array-like of shape (n,)
        Integer sense labels
    label_names : dict, optional
        Label ID to display name mapping {label_id: "Sense Name"}
    palette : dict, optional
        Color mapping {label_id: hex_color}
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height); only used when ``ax`` is None
    ax : matplotlib 3D Axes, optional
        Existing 3D axes to draw into. When None, a new standalone figure is
        created (original behaviour). When provided, ``figsize`` is ignored.
    show_legend : bool
        Whether to draw the axes legend. Set to False when composing this panel
        into a shared figure with a single legend.
    elev, azim : float
        Elevation and azimuth view angles passed to ``ax.view_init``.
    xlabel, ylabel, zlabel : str
        Axis labels.
        
    Returns
    -------
    fig, ax : matplotlib 3D figure and axes
        
    Notes
    -----
    - X, Y = the supplied 2D projection axes
    - Z = time (random_date values in years)
    - Points colored by sense label using provided or auto-generated palette
    """
    
    # Validate inputs
    if X_lda.shape[1] != 2:
        raise ValueError(f"X_lda must have 2 columns, got {X_lda.shape[1]}")
    if len(X_lda) != len(dates) or len(X_lda) != len(labels):
        raise ValueError(f"Input lengths mismatch: X_lda={len(X_lda)}, dates={len(dates)}, labels={len(labels)}")
    
    n_classes = len(np.unique(labels))
    
    # Generate palette if not provided
    if palette is None:
        palette = create_label_palette(n_classes)
    
    # Generate label names if not provided
    if label_names is None:
        label_names = {i: f"Sense {i}" for i in range(n_classes)}
    
    # Convert dates to numpy array for plotting
    dates_arr = np.asarray(dates)
    
    # Get colors for each point based on labels
    colors = [palette.get(l, "#cccccc") for l in labels]
    
    # Create figure and 3D axis
    if ax is None:
        ax_owned = True
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax_owned = False
        fig = ax.figure
    
    # Create scatter plot
    scatter = ax.scatter(
        X_lda[:, 0],           # projection dim 1
        X_lda[:, 1],           # projection dim 2
        dates_arr,             # Z = time (years)
        c=colors,
        s=30,
        alpha=0.6,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_zlabel(zlabel, fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set Z-axis limits with padding for temporal axis
    z_min, z_max = dates_arr.min(), dates_arr.max()
    z_padding = (z_max - z_min) * 0.05  # 5% padding
    ax.set_zlim(z_min - z_padding, z_max + z_padding)
    
    # Create legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               label=label_names.get(lid, f"Sense {lid}"),
               markerfacecolor=palette.get(lid, "#cccccc"),
               markersize=8)
        for lid in sorted(np.unique(labels))
    ]
    
    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=9, 
                  title="Sense Classes", title_fontsize=10)
    
    # Adjust view angle for better 3D visualization
    ax.view_init(elev=elev, azim=azim)
    
    if ax_owned:
        plt.tight_layout()
    return fig, ax
