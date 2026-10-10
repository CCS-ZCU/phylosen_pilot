"""Shared, label-free embedding ablations for 8_embedding-denoising.ipynb."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.extmath import randomized_svd


def unit_rows(X):
    X = np.asarray(X, dtype=float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    kind: str = "raw"
    count: int = 0
    strength: float = 1.0
    normalize: bool = False


class SharedFilterBank:
    """Fit one origin and nuisance candidates on training rows of both words.

    Equal total weight per word; no sense labels or cross-word correspondences.
    Top PCs are candidates, not established nuisance directions. Removing the
    word-mean direction is an explicit sensitivity experiment only.
    """

    def __init__(self, max_components=64, seed=42):
        self.max_components = max_components
        self.seed = seed

    def fit(self, X, words):
        X, words = np.asarray(X, dtype=float), np.asarray(words)
        if X.ndim != 2 or len(X) != len(words) or not np.isfinite(X).all():
            raise ValueError("Provide finite embeddings aligned with word identities.")
        terms, counts = np.unique(words, return_counts=True)
        if len(terms) != 2:
            raise ValueError("This experiment expects two words.")
        self.n_features_ = X.shape[1]
        weights = np.zeros(len(X))
        means = []
        for term, count in zip(terms, counts):
            weights[words == term] = 1 / (len(terms) * count)
            means.append(X[words == term].mean(axis=0))
        self.mean_ = np.mean(means, axis=0)
        centered = X - self.mean_
        self.coordinate_variance_ = np.sum(weights[:, None] * centered ** 2, axis=0)
        self.coordinate_order_ = np.argsort(-self.coordinate_variance_, kind="stable")
        weighted = centered * np.sqrt(weights[:, None])
        k = min(self.max_components, min(X.shape) - 1)
        _, values, self.components_ = randomized_svd(
            weighted, n_components=k, n_iter=7, random_state=self.seed)
        self.explained_variance_ratio_ = values ** 2 / max(
            float(self.coordinate_variance_.sum()), 1e-12)
        difference = means[0] - means[1]
        self.word_direction_ = difference / max(np.linalg.norm(difference), 1e-12)
        return self

    def transform(self, X, spec):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or not np.isfinite(X).all():
            raise ValueError("Input must match the fitted finite embedding matrix.")
        if not 0 <= spec.strength <= 1:
            raise ValueError("Attenuation strength must lie in [0, 1].")
        if spec.kind == "raw":
            result = X.copy()
        else:
            result = X - self.mean_
            if spec.kind == "pc":
                if not 0 <= spec.count <= len(self.components_):
                    raise ValueError("Requested PC count exceeds the fitted bank.")
                directions = self.components_[:spec.count]
                result = result - spec.strength * (result @ directions.T) @ directions
            elif spec.kind == "coordinate":
                if not 0 <= spec.count <= self.n_features_:
                    raise ValueError("Requested coordinate count exceeds embedding size.")
                result[:, self.coordinate_order_[:spec.count]] *= 1 - spec.strength
            elif spec.kind == "word_direction":
                direction = self.word_direction_
                result = result - spec.strength * np.outer(result @ direction, direction)
            elif spec.kind != "center":
                raise ValueError(f"Unknown filter kind: {spec.kind}")
        return unit_rows(result) if spec.normalize else result


def century_centroids(df, Z, min_count=3, bin_years=100):
    """Average actual usages in bins of one or several centuries.

    A width of 200 maps century midpoints 50/150 to 100 and -150/-50 to
    -100. Each usage has equal weight; existing century dates are preserved.
    """
    if (not isinstance(bin_years, (int, np.integer)) or bin_years < 100
            or bin_years % 100 != 0):
        raise ValueError("Bin width must be a positive multiple of 100 years.")
    Z = np.asarray(Z)
    if len(df) != len(Z) or Z.ndim != 2 or Z.shape[1] != 2 or min_count < 1:
        raise ValueError("Expected aligned 2-D coordinates and a positive count.")
    centuries = pd.to_numeric(df['century'], errors='coerce').to_numpy(float)
    valid = np.isfinite(centuries) & np.isfinite(Z).all(axis=1)
    d = df.loc[valid, ['word', 'sense']].reset_index(drop=True)
    midpoints = np.floor(centuries[valid] / bin_years) * bin_years + bin_years / 2
    d['year'], d['x'], d['y'] = midpoints, Z[valid, 0], Z[valid, 1]
    agg = dict(x=('x', 'mean'), y=('y', 'mean'), n=('x', 'size'))
    if 'grela_id' in df:
        d['work'] = df.loc[valid, 'grela_id'].to_numpy()
        agg['n_works'] = ('work', 'nunique')
    out = d.groupby(['word', 'sense', 'year'], as_index=False).agg(**agg)
    out['valid'] = out['n'] >= min_count
    return out.sort_values(['word', 'sense', 'year'])


def cross_word_distances(X, df, names):
    """Descriptive cosine distances between all cross-word sense means.

    No matching or alignment is used. Means are computed in the supplied space.
    """
    terms = sorted(df.word.unique())
    if len(terms) != 2:
        raise ValueError("Expected two words.")
    matrices, labels = [], []
    for term in terms:
        senses = sorted(df.loc[df.word == term, 'sense'].unique())
        matrices.append(np.stack([
            X[((df.word == term) & (df.sense == s)).to_numpy()].mean(axis=0)
            for s in senses]))
        labels.append([f'{term}: {names[term].get(int(s), str(s))}' for s in senses])
    return pd.DataFrame(cosine_distances(*matrices), index=labels[0], columns=labels[1])
