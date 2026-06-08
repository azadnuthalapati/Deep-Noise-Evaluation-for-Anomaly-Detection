import argparse
import time
import numpy as np
import torch

from noise import generate_noise, NOISE_TYPES
from model import build_model
from evaluate import evaluate_all


# ---------- data ----------

def load_npz(path):
    # ADBench .npz: X (n, d), y (n,) with 1 = anomaly
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float64)
    y = np.asarray(d["y"]).astype(int).ravel()
    return X, y


def make_synthetic(d=16, n_normal=2000, n_anom=300, seed=0):
    # small built-in dataset so the repo runs with no download
    rng = np.random.default_rng(seed)
    mean = rng.standard_normal(d)
    A = rng.standard_normal((d, d)) * 0.6
    cov = A @ A.T + np.eye(d) * 0.1
    normal = rng.multivariate_normal(mean, cov, size=n_normal)
    easy = rng.multivariate_normal(mean, cov * 6.0, size=n_anom // 2)
    hard = normal[: n_anom - n_anom // 2] + rng.standard_normal((n_anom - n_anom // 2, d)) * 1.2
    X = np.vstack([normal, easy, hard])
    y = np.concatenate([np.zeros(n_normal), np.ones(len(easy) + len(hard))]).astype(int)
    return X, y


def split_and_standardize(X, y, seed=0):
    # unsupervised protocol: train on half the normals, test on the rest + all anomalies
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    anom_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    cut = len(normal_idx) // 2
    tr_idx = normal_idx[:cut]
    te_idx = np.concatenate([normal_idx[cut:], anom_idx])
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_te = np.concatenate([np.zeros(len(normal_idx) - cut), np.ones(len(anom_idx))]).astype(int)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8       # standardize with train stats only
    return (X_tr - mu) / sd, (X_te - mu) / sd, y_te


# ---------- training  ----------

def train(X_tr, args, device):
    n, d = X_tr.shape
    model = build_model(d, args.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_decay_epoch], gamma=0.1)

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    mse = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device, generator=gen)
        running = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            B = Xt[idx]
            opt.zero_grad()
            loss = mse(model(B), torch.zeros_like(B))                 # clean -> 0
            for r in args.ratios:                                     # one noised copy per ratio
                E = generate_noise(B.shape[0], d, args.sigma_max, args.m,
                                   args.noise_type, r, device=device, generator=gen)
                loss = loss + mse(model(B + E), E.abs())               # noised -> |eps|
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={running / n:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help=".npz path; omit for synthetic")
    p.add_argument("--arch", default="mlp", choices=["mlp", "resmlp"])
    p.add_argument("--noise-type", dest="noise_type", default="gaussian", choices=list(NOISE_TYPES))
    p.add_argument("--sigma-max", dest="sigma_max", type=float, default=2.0)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--lr-decay-epoch", dest="lr_decay_epoch", type=int, default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default=None, help="optional checkpoint path")
    p.add_argument("--log-every", dest="log_every", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y = load_npz(args.data) if args.data else make_synthetic(seed=args.seed)
    X_tr, X_te, y_te = split_and_standardize(X, y, seed=args.seed)
    print(f"data: n_train={len(X_tr)}  n_test={len(X_te)}  d={X_tr.shape[1]}  "
          f"anomaly_frac={y_te.mean():.3f}  device={args.device}")

    t0 = time.time()
    model = train(X_tr, args, args.device)
    secs = time.time() - t0

    res = evaluate_all(model, X_tr, X_te, y_te, args.device)
    print(f"\n--- results ({args.arch}, {args.noise_type}, sigma_max={args.sigma_max}, m={args.m}) ---")
    for k, v in res.items():
        print(f"{k:22s} AUC = {v:.4f}")
    print(f"wall-clock train time: {secs:.1f}s")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "arch": args.arch}, args.save)
        print(f"saved checkpoint -> {args.save}")


if __name__ == "__main__":
    main()
