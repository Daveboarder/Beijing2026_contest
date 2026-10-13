import sys; sys.path.insert(0, "scripts"); import _bootstrap  # noqa
import numpy as np, pandas as pd, time, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from libs2026.config import Config
from libs2026.depth_transformer import build_depth_sequences
from libs2026.bimamba import BiMambaClassifier, augment_batch

cfg = Config.load(None)
X, index = build_depth_sequences(cfg, bin_factor=4, surface_shots=20, late_bin=4)
y = index["label"].to_numpy(int) - 1; ids = index["sample_id"].to_numpy()
folds = pd.read_csv("results/bimamba/initial/folds.csv").query("repeat==0")
pos = {s: i for i, s in enumerate(ids)}
EPOCHS, EVERY = int(sys.argv[2]), 10
dev = "cuda"
NORM = sys.argv[3]

def predict(net, x):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(torch.from_numpy(x[i:i+32]).to(dev)).argmax(1).cpu()
                          for i in range(0, len(x), 32)]).numpy()

for spec in sys.argv[1].split(","):
    variant, lr = spec.split(":"); lr = float(lr)
    test_pred = {e: np.zeros(len(y), int) for e in range(EVERY, EPOCHS + 1, EVERY)}
    train_acc = {e: [] for e in test_pred}
    t0 = time.time()
    for f in range(5):
        te = np.array([pos[s] for s in folds.query("fold==@f").sample_id])
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = BiMambaClassifier(channel_widths=(1023, 1023, 1023), variant=variant, device=dev, random_state=42)
        m.classes_ = np.arange(5)
        if NORM == "col":
            mean = X[tr].mean(axis=(0, 1)); std = np.maximum(X[tr].std(axis=(0, 1)), 1e-6)
        else:
            mean, std = m._stats(X[tr])
        xtr, xte = (X[tr] - mean) / std, (X[te] - mean) / std
        net = m._net()
        loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(y[tr])),
                            batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42))
        counts = np.bincount(y[tr], minlength=5)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (5 * counts), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        for epoch in range(1, EPOCHS + 1):
            net.train()
            for xb, yb in loader:
                xb, yb = augment_batch(xb.to(dev), 0.05, 0.1), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                out = net.outputs(xb)
                loss = crit(out["final"], yb)
                if "spectral" in out and "depth" in out:
                    loss = loss + 0.3 * (crit(out["spectral"], yb) + crit(out["depth"], yb))
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if epoch % EVERY == 0:
                test_pred[epoch][te] = predict(net, xte)
                train_acc[epoch].append((predict(net, xtr) == y[tr]).mean())
    line = " ".join(f"{e}:{np.mean(train_acc[e]):.2f}/{(test_pred[e]==y).mean():.2f}" for e in test_pred)
    print(f"{NORM:6s} {variant:8s} lr={lr:g} [epoch:train/test] {line}  ({time.time()-t0:.0f}s)", flush=True)
