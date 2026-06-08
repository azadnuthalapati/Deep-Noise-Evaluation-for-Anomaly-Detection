import math
import torch

NOISE_TYPES = ("gaussian", "laplace", "uniform", "rayleigh")


def _unit_noise(noise_type, shape, device, generator):
    # sample noise with mean 0, std 1; we scale by the per-feature sigma later
    if noise_type == "gaussian":
        return torch.randn(shape, device=device, generator=generator)
    if noise_type == "laplace":
        # Laplace(0, 1/sqrt(2)) has std 1; inverse-CDF sampling
        b = 1.0 / math.sqrt(2.0)
        u = torch.rand(shape, device=device, generator=generator) - 0.5
        return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs())
    if noise_type == "uniform":
        # U(-sqrt(3), sqrt(3)) has std 1
        a = math.sqrt(3.0)
        return torch.empty(shape, device=device).uniform_(-a, a, generator=generator)
    if noise_type == "rayleigh":
        # Rayleigh, then center and rescale to mean 0, std 1
        u = torch.rand(shape, device=device, generator=generator).clamp_(1e-7, 1 - 1e-7)
        r = torch.sqrt(-2.0 * torch.log(u))
        mean = math.sqrt(math.pi / 2.0)
        std = math.sqrt((4.0 - math.pi) / 2.0)
        return (r - mean) / std
    raise ValueError(f"unknown noise_type {noise_type}")


def generate_noise(batch_size, dim, sigma_max=2.0, m=3, noise_type="gaussian",
                   noise_ratio=1.0, device="cpu", generator=None):
    """Algorithm 1. Give each sample m different noise levels so that, across the
    batch, the noise spans the whole range [0, sigma_max]. That coverage is the
    diversity the paper's guarantee relies on."""
    device = torch.device(device)

    # split [0, sigma_max] into m bins; draw one level per (sample, bin)
    edges = torch.linspace(0.0, sigma_max, m + 1, device=device)
    u = torch.rand(batch_size, m, device=device, generator=generator)
    sigmas = edges[:-1] + u * (edges[1:] - edges[:-1])              # (batch, m)

    # put each feature in a bin (contiguous chunks), then shuffle the positions
    feat_bin = (torch.arange(dim, device=device) * m // dim).clamp_(max=m - 1)
    sigma = sigmas[:, feat_bin]                                     # (batch, dim)
    perm = torch.argsort(torch.rand(batch_size, dim, device=device, generator=generator), dim=1)
    sigma = torch.gather(sigma, 1, perm)

    E = _unit_noise(noise_type, (batch_size, dim), device, generator) * sigma

    # optionally noise only a fraction of the features
    if noise_ratio < 1.0:
        keep = torch.rand(batch_size, dim, device=device, generator=generator) < noise_ratio
        E = E * keep
    return E
