import sys; sys.path.insert(0, "scripts"); import _bootstrap  # noqa
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences
from libs2026.bimamba import depth_geometry
cfg = Config.load(None)
X, index = build_depth_sequences(cfg, bin_factor=4, surface_shots=20, late_bin=4)
y = index["label"].to_numpy(int); ids = index["sample_id"].to_numpy()
print("X", X.shape)
_, W = depth_geometry()
regions = np.einsum("rl,blf->brf", W, X)            # (120, 3, 3072)
feats = {
    "regions_flat (3x3072)": regions.reshape(len(X), -1),
    "surface_region": regions[:, 0],
    "aged_region": regions[:, 1],
    "bulk_region": regions[:, 2],
    "all_tokens_mean": X.mean(1),
    "first20_tokens_flat": X[:, :20].reshape(len(X), -1),
}
folds = pd.read_csv("results/bimamba/initial/folds.csv")
pos = {s: i for i, s in enumerate(ids)}
def run(F, rep, fold):
    te = np.array([pos[s] for s in folds.query("repeat==@rep and fold==@fold").sample_id])
    tr = np.setdiff1d(np.arange(len(y)), te)
    m = Pipeline([("s", StandardScaler()), ("p", PCA(30, random_state=0)), ("s2", StandardScaler()),
                  ("c", MLPClassifier((128, 64), max_iter=2000, alpha=1e-3, random_state=0))])
    m.fit(F[tr], y[tr]); return rep, te, m.predict(F[te])
for name, F in feats.items():
    res = Parallel(n_jobs=20)(delayed(run)(F, r, f) for r in range(4) for f in range(5))
    accs = []
    for r in range(4):
        pred = np.zeros(len(y), int)
        for rr, te, p in res:
            if rr == r: pred[te] = p
        accs.append((pred == y).mean())
    print(f"{name:28s} acc {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
