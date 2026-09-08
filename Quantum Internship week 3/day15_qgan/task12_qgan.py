"""Day 15 — 1-qubit generator QGAN vs tiny classical discriminator.

Target: 1D two-mode mixture on [-1, 1]. Generator maps noise through a
1-qubit circuit; discriminator is a small torch MLP. Adversarial BCE
training with fixed seeds. All local.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

SEED = 42
EPOCHS = 300
BATCH = 32
LR = 0.01
N_SAMPLE = 1000

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def get_device() -> qml.devices.Device:
    """Local simulator, Lightning preferred."""
    try:
        return qml.device("lightning.qubit", wires=1)
    except Exception:
        return qml.device("default.qubit", wires=1)


def target_samples(n: int = N_SAMPLE, seed: int = SEED) -> np.ndarray:
    """Two-mode Gaussian mixture clipped to [-1, 1]."""
    rng = np.random.RandomState(seed)
    mix = rng.rand(n) < 0.5
    x = np.where(mix, rng.normal(-0.5, 0.15, n), rng.normal(0.5, 0.15, n))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def build_generator() -> tuple[qml.QNode, torch.Tensor]:
    """Return the 1-qubit generator QNode and its seed-fixed parameters."""
    dev = get_device()

    @qml.qnode(dev, interface="torch")
    def circuit(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        qml.RY(theta[0] * z + theta[1], wires=0)
        qml.RZ(theta[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = torch.tensor([1.0, 0.1, 0.2], dtype=torch.float32)
    return circuit, theta


class Discriminator(nn.Module):
    """Tiny 1D classifier MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        return self.net(x).squeeze(-1)


def train(seed: int = SEED, epochs: int = EPOCHS) -> dict:
    """Adversarial training; return losses and histogram snapshots."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen, theta = build_generator()
    theta = theta.clone().detach().requires_grad_(True)
    disc = Discriminator()
    opt_g = torch.optim.Adam([theta], lr=LR)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    real_pool = target_samples()
    snaps: dict = {}
    gloss_hist, dloss_hist = [], []
    with torch.no_grad():
        z0 = torch.from_numpy(np.random.uniform(-1, 1, N_SAMPLE).astype(np.float32))
        snaps["epoch0"] = torch.stack([gen(z, theta) for z in z0]).numpy()
    for ep in range(epochs):
        idx = np.random.choice(len(real_pool), BATCH, replace=False)
        real = torch.from_numpy(real_pool[idx]).unsqueeze(-1)
        z = torch.rand(BATCH) * 2 - 1
        with torch.no_grad():
            fake_det = torch.stack([gen(zi, theta) for zi in z])
        fake = (fake_det + torch.randn(BATCH) * 0.02).unsqueeze(-1)
        opt_d.zero_grad()
        loss_d = bce(disc(real), torch.ones(BATCH)) + bce(disc(fake), torch.zeros(BATCH))
        loss_d.backward()
        opt_d.step()
        z2 = torch.rand(BATCH) * 2 - 1
        opt_g.zero_grad()
        fake2 = torch.stack([gen(zi, theta) for zi in z2]).unsqueeze(-1)
        loss_g = bce(disc(fake2), torch.ones(BATCH))
        loss_g.backward()
        opt_g.step()
        gloss_hist.append(float(loss_g.detach()))
        dloss_hist.append(float(loss_d.detach()))
        if ep == epochs // 2:
            with torch.no_grad():
                zm = torch.from_numpy(
                    np.random.uniform(-1, 1, N_SAMPLE).astype(np.float32))
                snaps["mid"] = torch.stack([gen(zz, theta) for zz in zm]).numpy()
    with torch.no_grad():
        zf = torch.from_numpy(np.random.uniform(-1, 1, N_SAMPLE).astype(np.float32))
        snaps["final"] = torch.stack([gen(zz, theta) for zz in zf]).numpy()
    return {"snaps": snaps, "gloss": gloss_hist, "dloss": dloss_hist,
            "theta": theta.detach().numpy(), "target": real_pool}


def save_figures(res: dict, epochs: int = EPOCHS) -> None:
    """Target-vs-generated histograms (0/mid/final) + loss curves."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, key, title in [
        (axes[0], "epoch0", "Epoch 0"),
        (axes[1], "mid", f"Epoch {epochs // 2}"),
        (axes[2], "final", f"Epoch {epochs}"),
    ]:
        ax.hist(res["target"], bins=30, alpha=0.5, label="Target")
        ax.hist(np.clip(res["snaps"][key], -1, 1), bins=30, alpha=0.5,
                label="Generated")
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    fig.suptitle("Day 15: target vs generated histograms")
    fig.savefig(ASSETS / "day15_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(res["gloss"], label="Generator")
    ax.plot(res["dloss"], label="Discriminator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Day 15: adversarial loss curves")
    ax.legend()
    fig.savefig(ASSETS / "day15_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Train the QGAN, save figures, print final losses."""
    res = train()
    save_figures(res)
    print(f"final_gloss={res['gloss'][-1]:.4f} final_dloss={res['dloss'][-1]:.4f}")


if __name__ == "__main__":
    main()
