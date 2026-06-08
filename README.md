# Deep-Noise-Evaluation-for-Anomaly-Detection
From-scratch PyTorch implementation of "Unsupervised Anomaly Detection for Tabular Data Using Noise Evaluation" (Dai, Hwang &amp; Fan, AAAI 2025), with an ADBench reproduction.
[arXiv:2412.11461](https://arxiv.org/abs/2412.11461).

## Idea

Train a network `h(x): R^d -> R^d` to read off how much noise is in a point.
We only have normal data, so we make the noise ourselves: add diverse synthetic
noise `E` to clean points and train two targets — clean points map to `0`, noised
points map to `|E|` (the per-feature noise magnitude). At test time the anomaly
score is `max_j |h(x)_j|`: a normal point stays small, an anomaly spikes.

The noise generator (`noise.py`) is the key part. Following Algorithm 1, each
sample is given `m` different noise levels drawn across `[0, sigma_max]`, so the
batch covers the whole range. That diversity is what lets the network flag hard
anomalies it never saw in training.

## Files

- `noise.py` — the diverse-noise generator (Algorithm 1).
- `model.py` — `VanillaMLP` and `ResMLP`.
- `train.py` — training loop, data loading, CLI.
- `evaluate.py` — max-score AUC plus the IsolationForest / LOF baselines.

## Install & run

```bash
pip install torch numpy scikit-learn

# synthetic smoke test (no download)
python train.py

# ADBench datasets: download the .npz files from
# https://github.com/Minqi824/ADBench (adbench/datasets/Classical) into datasets/
python train.py --data datasets/6_cardio.npz     --arch mlp    --epochs 200
python train.py --data datasets/30_satellite.npz  --arch mlp    --epochs 200
python train.py --data datasets/26_optdigits.npz  --arch resmlp --epochs 200

# re-score a saved checkpoint
python train.py    --data datasets/6_cardio.npz --save runs/cardio.pt
python evaluate.py --data datasets/6_cardio.npz --ckpt runs/cardio.pt
```

Main flags: `--arch {mlp,resmlp}`, `--noise-type {gaussian,laplace,uniform,rayleigh}`,
`--sigma-max`, `--m`, `--epochs`, `--lr`, `--batch-size`, `--seed`.
Defaults follow the paper: Gaussian noise, `sigma_max=2`, `m=3`, three noised
copies per sample (ratios 0.5/0.8/1.0), Adam+AMSGrad `lr=1e-4`, `wd=5e-4`,
learning rate ×0.1 at epoch 100.

## Part 2 — results

AUC-ROC ×100, 200 epochs, seed 0. **Bold** = best of the three detectors on that
dataset. Run on an HP laptop, 16 GB RAM, CPU only.

| Dataset   | d  | % anom | Arch   | NoiseEval (mine) | IsolationForest | LocalOutlierFactor | Paper | Train time |
|-----------|:--:|:------:|:------:|:----------------:|:---------------:|:------------------:|:-----:|:----------:|
| cardio    | 21 |  9.6   | MLP    |    **96.19**     |      95.76      |       92.30        | 96.66 |   4.5 s    |
| satellite | 36 |  31.6  | MLP    |      78.56       |      81.31      |     **84.85**      | 82.06 |   14.5 s   |
| optdigits | 64 |  2.9   | ResMLP |      87.26       |      86.60      |     **97.12**      | 92.44 |   78.9 s   |

*Paper = the matching architecture's number from Table 6 (UAD setting).*

Cardio beats both baselines; optdigits edges out IsolationForest but loses to LOF;
satellite beats neither. That is the expected pattern — the paper also doesn't win
on satellite/optdigits in the unsupervised setting, and satellite's test split is
~48% anomalies, far from the "anomalies are rare noise" assumption the method needs.

Running 500 epochs instead of 200 changed almost nothing (cardio 95.99, satellite
78.25, optdigits 85.06 — all within run-to-run noise, two slightly lower), so the
method converges well before 500 and the gaps are the method, not under-training.

## Implementation choices

The paper leaves a few things open; my choices:

- **`m` is the number of feature chunks**, filled with contiguous levels then
  shuffled, so the levels scatter across the vector.
- **Output is `abs()`-ed before the max** instead of using a non-negative output
  layer, since the target `|E|` is already non-negative.
- **`sigma` is sampled at unit std and scaled**, which is exact for the
  location/scale noise families used here.

## Existing code

I did not find a public implementation from the authors. None was used.
