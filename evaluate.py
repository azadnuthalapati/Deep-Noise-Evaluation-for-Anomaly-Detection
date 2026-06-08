import argparse
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

from model import build_model


@torch.no_grad()
def score_model(model, X, device="cpu"):
    # anomaly score = max over feature dims of |output|  (Eq. 10)
    model.eval()
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    return model(xt).abs().max(dim=1).values.cpu().numpy()


def baseline_aucs(X_tr, X_te, y_te):
    out = {}
    iforest = IsolationForest(random_state=0).fit(X_tr)
    out["IsolationForest"] = roc_auc_score(y_te, -iforest.score_samples(X_te))
    lof = LocalOutlierFactor(novelty=True).fit(X_tr)
    out["LocalOutlierFactor"] = roc_auc_score(y_te, -lof.score_samples(X_te))
    return out


def evaluate_all(model, X_tr, X_te, y_te, device="cpu"):
    res = {"NoiseEval": roc_auc_score(y_te, score_model(model, X_te, device))}
    res.update(baseline_aucs(X_tr, X_te, y_te))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help=".npz path; omit for synthetic")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    from train import load_npz, make_synthetic, split_and_standardize
    X, y = load_npz(a.data) if a.data else make_synthetic(seed=a.seed)
    X_tr, X_te, y_te = split_and_standardize(X, y, seed=a.seed)

    ckpt = torch.load(a.ckpt, map_location=a.device)
    model = build_model(X_tr.shape[1], ckpt.get("arch", "mlp")).to(a.device)
    model.load_state_dict(ckpt["state_dict"])

    for k, v in evaluate_all(model, X_tr, X_te, y_te, a.device).items():
        print(f"{k:22s} AUC = {v:.4f}")


if __name__ == "__main__":
    main()
