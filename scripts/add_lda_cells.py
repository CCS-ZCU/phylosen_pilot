#!/usr/bin/env python3
"""
Safely add LDA and trajectory analysis cells to 4_projections.ipynb
using nbformat to preserve notebook structure.
"""

import nbformat
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "4_projections.ipynb"

# Define the new cells to add
NEW_CELLS = [
    # Cell 1: Import LDA tools
    {
        "type": "code",
        "source": """# =============================================================================
# SUPERVISED DIMENSIONALITY REDUCTION WITH LDA
# =============================================================================
from lda_projections import LDAProjection, compute_lda_projection, evaluate_lda_vs_unsupervised
print("LDA modules imported successfully!")"""
    },
    # Cell 2: Compare methods
    {
        "type": "code",
        "source": """# Prepare data for passio
X_passio = np.stack(passio_df["emb_philberta_l11"])
y_passio = passio_df["class_llama3.3"].dropna().astype(int).values
X_clean = X_passio[:len(y_passio)]

print("=== PASSIO: LDA vs Unsupervised Methods ===\\n")
results_passio = evaluate_lda_vs_unsupervised(X_clean, y_passio, REDUCTIONS)

for method, score in sorted(results_passio.items(), key=lambda x: x[1] if isinstance(x[1], float) else -999, reverse=True):
    print(f"{method:20s}: {score:.4f}" if isinstance(score, float) else f"{method:20s}: FAILED")

# Same for pathos
X_pathos = np.stack(pathos_df["emb_philberta_l11"])
y_pathos = pathos_df["class_llama3.3"].dropna().astype(int).values
X_pathos_clean = X_pathos[:len(y_pathos)]

print("\\n=== PATHOS: LDA vs Unsupervised Methods ===\\n")
results_pathos = evaluate_lda_vs_unsupervised(X_pathos_clean, y_pathos, REDUCTIONS)

for method, score in sorted(results_pathos.items(), key=lambda x: x[1] if isinstance(x[1], float) else -999, reverse=True):
    print(f"{method:20s}: {score:.4f}" if isinstance(score, float) else f"{method:20s}: FAILED")"""
    },
    # Cell 3: Visual comparison PCA vs LDA
    {
        "type": "code",
        "source": """from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Compute projections for passio
X_lda_passio, lda_passio = compute_lda_projection(X_clean, y_passio)
pca_passio = PCA(n_components=2, random_state=42)
X_pca_passio = pca_passio.fit_transform(X_clean)

# Scores
sil_lda_passio = silhouette_score(X_lda_passio, y_passio)
sil_pca_passio = silhouette_score(X_pca_passio, y_passio)
impr_passio = (sil_lda_passio - sil_pca_passio) / abs(sil_pca_passio) * 100 if sil_pca_passio != 0 else 0

print(f"PASSIO: PCA={sil_pca_passio:.4f}, LDA={sil_lda_passio:.4f} (+{impr_passio:.1f}%)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca_passio[:,0], X_pca_passio[:,1], c=y_passio, cmap='tab10', s=20, alpha=0.6)
axes[0].set_title(f"Passio PCA (sil={sil_pca_passio:.3f})", fontsize=12)
axes[0].axis('off')

axes[1].scatter(X_lda_passio[:,0], X_lda_passio[:,1], c=y_passio, cmap='tab10', s=20, alpha=0.6)
axes[1].set_title(f"Passio LDA (sil={sil_lda_passio:.3f})", fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Feature importance
print("\\nTop 10 discriminative dimensions for Passio:")
top_dims = lda_passio.get_top_dimensions(k=10)
importances = lda_passio.get_feature_importances()
for idx in top_dims:
    print(f"  Dimension {idx:3d}: importance={importances[idx]:.3f}")"""
    },
    # Cell 4: Centroid trajectories
    {
        "type": "code",
        "source": """# =============================================================================
# TEMPORAL TRAJECTORIES OF SENSE CENTROIDS
# =============================================================================
print("Computing sense centroids across historical epochs...")

# Define epochs
def assign_epoch(year):
    if pd.isna(year): return "Unknown"
    if year < 500: return "Antiquity"
    elif year < 1000: return "Late Ant./Medieval"
    elif year < 1500: return "High Medieval"
    else: return "Early Modern"

passio_df["epoch"] = passio_df["random_date"].apply(assign_epoch)
epochs = ["Antiquity", "Late Ant./Medieval", "High Medieval", "Early Modern"]

# Compute centroids for each epoch and sense
centroids = {}
for epoch in epochs:
    epoch_mask = passio_df["epoch"] == epoch
    if epoch_mask.sum() == 0:
        continue
    for label in np.unique(y_passio):
        class_mask = epoch_mask & (passio_df["class_llama3.3"] == label)
        if class_mask.sum() > 0:
            key = (epoch, int(label))
            centroids[key] = X_clean[class_mask.values].mean(axis=0)

print(f"Computed {len(centroids)} centroids for Passio")

# Plot trajectories in LDA space
fig, ax = plt.subplots(figsize=(12, 8))

for label in np.unique(y_passio):
    points = []
    epoch_names = []
    for epoch in epochs:
        key = (epoch, int(label))
        if key in centroids:
            p = lda_passio.transform(centroids[key].reshape(1, -1))[0]
            points.append(p)
            epoch_names.append(epoch)
    
    if len(points) > 1:
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], 'o-', 
                color=plt.cm.tab10(label), label=f"Sense {label}", 
                linewidth=2.5, markersize=12, alpha=0.9)
        # Annotate epochs
        for i, ep in enumerate(epoch_names):
            ax.annotate(ep.split('/')[-1][:4], (points[i, 0], points[i, 1]), 
                       fontsize=9, fontweight='bold', alpha=0.8,
                       ha='center', va='bottom')

ax.set_title("Passio: Sense Trajectories in LDA Space (Diachronic Evolution)", fontsize=16, pad=20)
ax.set_xlabel("LD1 (Discriminant Axis 1)", fontsize=12)
ax.set_ylabel("LD2 (Discriminant Axis 2)", fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

print("\\nTrajectories show how senses evolve over time in semantic space.")
print("Branching indicates sense splits; convergence indicates blending.")"""
    }
]

def add_cells_to_notebook():
    # Read existing notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Find insertion point (after imports/setup, before any heavy computation)
    # We'll insert after the first few import cells
    insert_index = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'REDUCTIONS' in ''.join(cell.source):
            insert_index = i + 1
            break
    
    # Create new cells
    new_cells = []
    for cell_data in NEW_CELLS:
        if cell_data['type'] == 'code':
            new_cell = nbformat.v4.new_code_cell(cell_data['source'])
        elif cell_data['type'] == 'markdown':
            new_cell = nbformat.v4.new_markdown_cell(cell_data['source'])
        new_cells.append(new_cell)
    
    # Insert cells
    nb.cells[insert_index:insert_index] = new_cells
    
    # Write back
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Successfully added {len(new_cells)} cells to {NOTEBOOK_PATH}")
    print(f"Inserted at position {insert_index}")
    print("You can now run these cells in your notebook!")

if __name__ == "__main__":
    add_cells_to_notebook()
