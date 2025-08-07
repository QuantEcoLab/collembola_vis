from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

df = pd.read_csv('data/templates/features_matrix.csv', index_col=0)

sim = cosine_similarity(df.values)
dist_spf = squareform(1 - sim, checks=False)

Z = linkage(dist_spf, method='complete')

labels = fcluster(Z, t=0.04, criterion='distance')

clusters = {}
for name, lbl in zip(df.index, labels):
    clusters.setdefault(lbl, []).append(name)

for i, members in enumerate([c for c in clusters.values() if len(c)>1], start=1):
    print(f"Klaster {i}:")
    for m in members:
        print("  -", m)