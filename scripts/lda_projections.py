"""
LDA-based supervised dimensionality reduction for semantic visualization.

This module provides tools to project high-dimensional embeddings (e.g., PhilBerta 768D)
into 2D space using Linear Discriminant Analysis, which maximizes class separation.

Unlike PCA/UMAP/t-SNE (unsupervised), LDA uses sense labels to find directions that
best distinguish between word senses - crucial for ERC proposal visualizations.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, Optional


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
