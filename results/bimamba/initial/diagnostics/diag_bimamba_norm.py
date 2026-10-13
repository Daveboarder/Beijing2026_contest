import sys; sys.path.insert(0, "scripts"); import _bootstrap  # noqa
import numpy as np, pandas as pd, time
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences
from libs2026.bimamba import BiMambaClassifier

class PerColumn(BiMambaClassifier):
    def _stats(self, x):
        mean = x.mean(axis=(0, 1)).astype(np.float32)
        std = np.maximum(x.std(axis=(0, 1)), 1e-6).astype(np.float32)
        return mean, std

cfg = Config.load(None)
X, index = build_depth_sequences(cfg, bin_factor=4, surface_shots=20, late_bin=4)
y = index["label"].to_numpy(int); ids = index["sample_id"].to_numpy()
folds = pd.read_csv("results/bimamba/initial/folds.csv").query("repeat==0")
pos = {s: i for i, s in enumerate(ids)}
variants = sys.argv[1].split(",")
for cls in (PerColumn, BiMambaClassifier):
    for variant in variants:
        pred = np.zeros(len(y), int); tracc = []; ep = []; t0 = time.time()
        for f in range(5):
            te = np.array([pos[s] for s in folds.query("fold==@f").sample_id])
            tr = np.setdiff1d(np.arange(len(y)), te)
            m = cls(channel_widths=(1023, 1023, 1023), variant=variant, device="cpu", random_state=42)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te]); tracc.append((m.predict(X[tr]) == y[tr]).mean()); ep.append(m.best_epoch_)
        print(f"{cls.__name__:18s} {variant:9s} test_acc {(pred==y).mean():.3f} train_acc {np.mean(tracc):.3f} "
              f"epochs {ep} {time.time()-t0:.0f}s", flush=True)
