"""
LDA-based supervised dimensionality reduction for semantic visualization.

This module provides tools to project high-dimensional embeddings (e.g., PhilBerta 768D)
into 2D space using Linear Discriminant Analysis, which maximizes class separation.

Unlike PCA/UMAP/t-SNE (unsupervised), LDA uses sense labels to find directions that
best distinguish between word senses - crucial for ERC proposal visualizations.
"""

import json
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, Optional, Union


class LDAProjection:
    """
    Wrapper for Linear Discriminant Analysis with convenient methods for
    semantic embedding visualization.
    
    Usage:
        >>> from lda_projections import LDAProjection
        >>> X = np.stack(df["emb_philberta_l11"])  # (n_samples, 768)
        >>> y = df["class_llama3.3"].astype(int).values
        >>> 
        >>> lda = LDAProjection(n_components=2)
        >>> X_2d = lda.fit_transform(X, y)
        >>> 
        >>> # Inspect important dimensions
        >>> top_dims = lda.get_top_dimensions(k=10)
        >>> print(f"Top discriminative dims: {top_dims}")
    """
    
    def __init__(self, n_components: int = 2, random_state: int = 42):
        """
        Initialize LDA projector.
        
        Parameters
        ----------
        n_components : int
            Number of discriminant axes (max is n_classes - 1)
        random_state : int
            For reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model: Optional[LinearDiscriminantAnalysis] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDAProjection':
        """
        Fit LDA on embeddings and labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Embeddings (e.g., 768-dim vectors)
        y : np.ndarray of shape (n_samples,)
            Sense labels (integers)
            
        Returns
        -------
        self
        """
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
        
        Returns
        -------
        importances : np.ndarray of shape (n_features,)
        """
        if self.model is None or not hasattr(self.model, 'scalings_'):
            raise ValueError("Model not fitted yet.")
        # scalings_: shape (n_features, n_components)
        return np.abs(self.model.scalings_).sum(axis=1)
    
    def get_top_dimensions(self, k: int = 10) -> np.ndarray:
        """
        Return indices of top-k most discriminative original dimensions.
        
        Parameters
        ----------
        k : int
            Number of top dimensions to return
            
        Returns
        -------
        top_indices : np.ndarray of shape (k,)
            Indices sorted by importance (descending)
        """
        importances = self.get_feature_importances()
        return np.argsort(importances)[-k:][::-1]


def compute_lda_projection(
    X: np.ndarray, 
    y: np.ndarray, 
    n_components: int = 2, 
    random_state: int = 42
) -> Tuple[np.ndarray, LDAProjection]:
    """
    Convenience function to compute LDA projection.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Embeddings (e.g., 768-dim PhilBerta vectors)
    y : np.ndarray of shape (n_samples,)
        Sense labels (integers)
    n_components : int
        Number of discriminant axes
    random_state : int
        
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


@dataclass
class SharedLDAProjection:
    """A pooled LDA model with the same selected axes for every lexeme."""

    model: LinearDiscriminantAnalysis
    axes: Tuple[int, int]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)[:, self.axes]


def compute_shared_lda_projection(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    axes: Tuple[int, int] = (1, 2),
) -> Tuple[Dict[str, np.ndarray], SharedLDAProjection]:
    """Fit one pooled LDA basis to multiple lexemes from the same encoder.

    Classes are distinct (lexeme, sense) pairs: sense 0 in one inventory is
    never equated with sense 0 in another. Each lexeme receives equal total
    prior weight, with empirical sense proportions within that lexeme.
    All embedding dimensions are retained; no per-dimension selection or
    per-lexeme centering is applied. Inputs must use the same encoder/layer.

    Returns per-lexeme coordinates and the *single* fitted model, which must
    also be used to project century centroids. This provides a common
    visualization, not evidence of cross-language alignment or ancestry.
    ``axes`` selects two zero-based discriminants. The default LD2/LD3 omits
    LD1, which strongly separates lexemes in the passio/pathos pilot; choose
    (0, 1) to inspect that separation. SVD uses all embedding dimensions and
    avoids the poorly separating auto-shrinkage axes observed in this pilot.
    """
    if len(embeddings) < 2 or set(embeddings) != set(labels):
        raise ValueError("Provide matching embedding and label keys for at least two lexemes.")

    matrices, joint_labels, priors = [], [], []
    feature_count = None
    next_class = 0
    for lexeme, values in embeddings.items():
        X = np.asarray(values, dtype=float)
        y = np.asarray(labels[lexeme])
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"{lexeme}: expected a nonempty matrix aligned with 1D labels.")
        if not np.isfinite(X).all():
            raise ValueError(f"{lexeme}: embeddings must be finite.")
        if feature_count is not None and X.shape[1] != feature_count:
            raise ValueError("All lexemes must use the same embedding dimensions.")
        feature_count = X.shape[1]
        _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        if np.any(counts < 2):
            raise ValueError(f"{lexeme}: each sense needs at least two observations.")
        matrices.append(X)
        joint_labels.append(inverse + next_class)
        priors.extend(counts / len(y) / len(embeddings))
        next_class += len(counts)

    if (len(axes) != 2 or len(set(axes)) != 2 or
            any(not isinstance(axis, (int, np.integer)) or axis < 0
                or axis >= min(feature_count, next_class - 1) for axis in axes)):
        raise ValueError("axes must select two distinct available discriminants (zero-based).")
    model = LinearDiscriminantAnalysis(
        n_components=max(axes) + 1, solver="svd",
        priors=np.asarray(priors),
    )
    model.fit(np.vstack(matrices), np.concatenate(joint_labels))
    shared = SharedLDAProjection(model=model, axes=tuple(axes))
    projected = {name: shared.transform(X) for name, X in embeddings.items()}
    return projected, shared


@dataclass
class WithinLexemeSharedProjection:
    """One linear map assembled from independent within-lexeme LDA models."""

    directions_: np.ndarray
    rotation_: np.ndarray
    origin_: np.ndarray
    local_models_: Dict[str, LinearDiscriminantAnalysis]
    explained_variance_ratio_: np.ndarray

    def transform_discriminant_space(self, X: np.ndarray) -> np.ndarray:
        """Retain every combined discriminant score, before the display reduction."""
        return (np.asarray(X, dtype=float) - self.origin_) @ self.directions_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform_discriminant_space(X) @ self.rotation_


def compute_within_lexeme_shared_projection(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    n_components: int = 2,
) -> Tuple[Dict[str, np.ndarray], WithinLexemeSharedProjection]:
    """Combine local LDA directions without cross-lexeme sense correspondences.

    Fit each lexeme's LDA on its own inventory. Concatenate its full discriminant
    loading matrix with those of the other lexemes, normalizing each matrix by
    its Frobenius norm to give each lexeme equal loading energy. Every input is
    then scored against ALL directions, irrespective of which word it contains.

    For display, diagonalize the equally weighted within-lexeme covariance of
    these common scores. This is an unsupervised rotation/reduction of the
    supervised common scores: it avoids choosing display axes merely because
    word means differ. Lexeme means are NOT removed at transformation time;
    both words share one origin and one linear map. Residual language offsets
    can remain, and proximity is not a validated bilingual alignment.
    """
    if len(embeddings) < 2 or set(embeddings) != set(labels):
        raise ValueError("Provide matching embeddings and labels for at least two lexemes.")
    matrices, local_models, blocks = {}, {}, []
    feature_count = None
    for term, values in embeddings.items():
        X, y = np.asarray(values, dtype=float), np.asarray(labels[term])
        if (X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(X)
                or not np.isfinite(X).all()):
            raise ValueError(f"{term}: provide a finite matrix aligned with 1D labels.")
        if feature_count is not None and X.shape[1] != feature_count:
            raise ValueError("All inputs must use the same embedding dimensions and order.")
        feature_count = X.shape[1]
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or np.any(counts < 2) or len(X) <= len(classes):
            raise ValueError(f"{term}: need two senses, two rows per sense and residual observations.")
        local = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
        directions = local.scalings_[:, :len(local.explained_variance_ratio_)]
        energy = np.linalg.norm(directions)
        if not np.isfinite(energy) or energy == 0:
            raise ValueError(f"{term}: no usable discriminant directions.")
        blocks.append(directions / energy)
        matrices[term], local_models[term] = X, local
    basis = np.concatenate(blocks, axis=1)
    covariance = np.zeros((basis.shape[1], basis.shape[1]))
    for X in matrices.values():
        scores = (X - X.mean(axis=0)) @ basis
        covariance += scores.T @ scores / (len(X) - 1) / len(matrices)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    available = np.sum(values > values[0] * 1e-12) if values[0] > 0 else 0
    if (not isinstance(n_components, (int, np.integer)) or n_components < 1
            or n_components > available):
        raise ValueError(f"n_components must be between 1 and {available}.")
    origin = np.mean([X.mean(axis=0) for X in matrices.values()], axis=0)
    shared = WithinLexemeSharedProjection(
        directions_=basis, rotation_=vectors[:, :n_components], origin_=origin,
        local_models_=local_models, explained_variance_ratio_=values / values.sum(),
    )
    return {term: shared.transform(X) for term, X in matrices.items()}, shared


def evaluate_lda_vs_unsupervised(
    X: np.ndarray, 
    y: np.ndarray,
    unsup_methods: Dict[str, callable] = None
) -> Dict[str, float]:
    """
    Compare LDA against unsupervised methods using silhouette scores.
    
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
    if unsup_methods is None:
        unsup_methods = {}
    
    results = {}
    
    # LDA
    X_lda, _ = compute_lda_projection(X, y, n_components=2)
    results['LDA'] = silhouette_score(X_lda, y)
    
    # Unsupervised methods
    for name, reducer_fn in unsup_methods.items():
        try:
            reducer = reducer_fn()
            if hasattr(reducer, 'fit_transform'):
                X_2d = reducer.fit_transform(X)
            else:
                X_2d = reducer(X)
            results[name] = silhouette_score(X_2d, y)
        except Exception as e:
            results[name] = float('-inf')  # Mark as failed
    
    return results


def compare_projections(
    X: np.ndarray,
    y: np.ndarray,
    red_func: callable,
    red_name: str = "Unsupervised"
) -> Tuple[float, float]:
    """
    Quick comparison of LDA vs a single unsupervised method.
    
    Returns
    -------
    (silhouette_lda, silhouette_unsup)
    """
    X_lda, _ = compute_lda_projection(X, y)
    sil_lda = silhouette_score(X_lda, y)
    
    reducer = red_func()
    if hasattr(reducer, 'fit_transform'):
        X_unsup = reducer.fit_transform(X)
    else:
        X_unsup = reducer(X)
    sil_unsup = silhouette_score(X_unsup, y)
    
    return sil_lda, sil_unsup


# =============================================================================
# Hybrid discriminative-dimension selection (cross-corpus)
# =============================================================================
#
# Motivation
# ----------
# LDA fitted separately on two lexemes (e.g. passio and pathos) each provides a
# ranking of the *original* embedding dimensions by how much they contribute to
# separating that lexeme's senses. Combining the two rankings yields a compact,
# sense-aware subspace that is informative for *both* lexemes.
#
# By default we take the **union** of each corpus' top-k dims ("include both"),
# which keeps every discriminative signal and guarantees at most 2*k dimensions;
# the intersection is still reported as a diagnostic.

def compute_top_dims(lda_model: Union['LDAProjection', LinearDiscriminantAnalysis],
                     k: int = 10) -> np.ndarray:
    """
    Return indices of the top-k most discriminative original dimensions.

    Accepts either a fitted :class:`LDAProjection` wrapper or a raw
    ``sklearn`` ``LinearDiscriminantAnalysis`` instance.

    Parameters
    ----------
    lda_model : LDAProjection or LinearDiscriminantAnalysis
        A fitted LDA model (must expose ``scalings_``).
    k : int
        Number of dimensions to return (capped at ``n_features``).

    Returns
    -------
    top_indices : np.ndarray, shape (k,)
        Dimension indices sorted by importance (descending).
    """
    model = getattr(lda_model, 'model', lda_model)
    if model is None or not hasattr(model, 'scalings_'):
        raise ValueError("LDA model is not fitted (missing 'scalings_').")
    importances = np.abs(model.scalings_).sum(axis=1)
    k = min(k, len(importances))
    return np.argsort(importances)[-k:][::-1]


@dataclass
class HybridDimSelection:
    """
    Result of combining per-corpus LDA discriminative dimension rankings.

    Attributes
    ----------
    dims : np.ndarray
        The selected hybrid dimensions (sorted ascending). Equal to ``union``
        when ``mode='union'`` and to ``intersection`` when ``mode='intersection'``.
    topk : dict[str, np.ndarray]
        Per-corpus top-k indices (ranked, descending importance).
    intersection : np.ndarray
        Dimensions shared by every corpus' top-k set.
    union : np.ndarray
        Dimensions appearing in at least one corpus' top-k set.
    k : int
        The ``k`` used per corpus.
    mode : str
        ``'union'`` or ``'intersection'``.
    """
    dims: np.ndarray
    topk: Dict[str, np.ndarray]
    intersection: np.ndarray
    union: np.ndarray
    k: int
    mode: str

    def __len__(self) -> int:
        return len(self.dims)

    def as_dict(self) -> Dict:
        """JSON-serialisable representation."""
        return {
            "dims": [int(d) for d in self.dims],
            "topk": {c: [int(d) for d in v] for c, v in self.topk.items()},
            "intersection": [int(d) for d in self.intersection],
            "union": [int(d) for d in self.union],
            "k": int(self.k),
            "mode": self.mode,
        }


def compute_hybrid_dims(lda_models: Dict[str, object],
                        k: int = 10,
                        mode: str = "union") -> HybridDimSelection:
    """
    Combine per-corpus LDA rankings into a hybrid dimension set.

    Parameters
    ----------
    lda_models : dict[str, LDAProjection | LinearDiscriminantAnalysis]
        Mapping ``{corpus_name: fitted_lda}`` (at least two corpora).
    k : int
        Number of top dimensions taken from each corpus.
    mode : {'union', 'intersection'}
        ``'union'`` (default) keeps dims discriminative for *any* corpus;
        ``'intersection'`` keeps only dims discriminative for *all* corpora.

    Returns
    -------
    HybridDimSelection
    """
    if mode not in ("union", "intersection"):
        raise ValueError("mode must be 'union' or 'intersection'")
    if len(lda_models) < 2:
        raise ValueError("Need at least two LDA models to combine.")

    topk = {name: compute_top_dims(model, k=k) for name, model in lda_models.items()}
    sets = [set(v.tolist()) for v in topk.values()]

    union = np.array(sorted(set().union(*sets)), dtype=int)
    inter = np.array(sorted(set.intersection(*sets)), dtype=int)
    dims = union if mode == "union" else inter

    return HybridDimSelection(
        dims=dims, topk=topk, intersection=inter, union=union, k=k, mode=mode
    )


def select_dims(X: np.ndarray, dims) -> np.ndarray:
    """Subselect columns ``dims`` from an embedding matrix ``X``."""
    dims = np.asarray(dims, dtype=int)
    return X[:, dims]


def save_hybrid_dims(selection: HybridDimSelection, path, meta: Optional[Dict] = None) -> None:
    """Persist a :class:`HybridDimSelection` (plus optional metadata) to JSON."""
    payload = selection.as_dict()
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
