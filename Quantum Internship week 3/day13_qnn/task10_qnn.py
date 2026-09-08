"""Day 13 — multi-qubit variational classifier with layer-depth sweep.

Depths 1-4 on a train/val moons split. Small-variance identity-like weight
initialization keeps early circuits shallow-effective and avoids barren
plateaus at these widths. All local, fixed seeds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 42
N_QUBITS = 2
DEPTHS = (1, 2, 3, 4)
EPOCHS = 40
LR = 0.05
INIT_STD = 0.1

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def get_device() -> qml.devices.Device:
    """Local simulator, Lightning preferred."""
    try:
        return qml.device("lightning.qubit", wires=N_QUBITS)
    except Exception:
        return qml.device("default.qubit", wires=N_QUBITS)


def build_qnode(dev: qml.devices.Device, depth: int) -> qml.QNode:
    """Variational classifier QNode with fixed depth."""

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    circuit._depth = depth  # type: ignore[attr-defined]
    return circuit


class VarClassifier(nn.Module):
    """AngleEmbedding + fixed-depth entangler + bias logit."""

    def __init__(self, qnode: qml.QNode, depth: int) -> None:
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (depth, N_QUBITS, 3)})
        with torch.no_grad():
            self.qlayer.weights.data.normal_(0.0, INIT_STD)  # near-identity init
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.qlayer(x) + self.bias).squeeze(-1)


def load_split(seed: int = SEED) -> tuple:
    """Moons train/val split as float32 torch tensors."""
    X, y = make_moons(n_samples=160, noise=0.15, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    conv = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    return conv(Xtr), conv(Xva), conv(ytr), conv(yva), Xtr, Xva, ytr, yva


def train_depth(depth: int, seed: int = SEED) -> dict:
    """Train one depth; return per-epoch train/val accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device()
    model = VarClassifier(build_qnode(dev, depth), depth)
    Xtr, Xva, ytr, yva, _, _, _, _ = load_split(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    tr_hist, va_hist = [], []
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            tr = float((((model(Xtr) > 0).float() == ytr).float()).mean())
            va = float((((model(Xva) > 0).float() == yva).float()).mean())
        tr_hist.append(tr)
        va_hist.append(va)
    return {"depth": depth, "model": model, "train": tr_hist, "val": va_hist,
            "final_val": va_hist[-1]}


def save_figures(results: list[dict]) -> None:
    """Accuracy-vs-epochs per depth + best-depth decision regions."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for r in results:
        ax.plot(r["val"], label=f"depth {r['depth']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Day 13: accuracy vs epochs per depth")
    ax.legend()
    fig.savefig(ASSETS / "day13_accuracy_vs_epochs.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    best = max(results, key=lambda r: r["final_val"])
    _, _, _, _, Xtr, Xva, ytr, yva = load_split()
    model = best["model"]
    model.eval()
    h = 0.08
    x0 = np.arange(-1.6, 2.6, h)
    x1 = np.arange(-1.2, 1.6, h)
    xx, yy = np.meshgrid(x0, x1)
    with torch.no_grad():
        grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
        zz = (model(grid) > 0).float().numpy().reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, zz, alpha=0.3)
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, edgecolors="k", s=14, label="Train")
    ax.scatter(Xva[:, 0], Xva[:, 1], c=yva, marker="s", edgecolors="k",
               s=14, label="Val")
    ax.set_title(f"Day 13: decision regions (depth {best['depth']})")
    ax.legend()
    fig.savefig(ASSETS / "day13_decision_regions.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Sweep depths, save figures, print accuracy table."""
    results = [train_depth(d) for d in DEPTHS]
    save_figures(results)
    for r in results:
        print(f"depth={r['depth']} val_acc={r['final_val']:.4f}")


if __name__ == "__main__":
    main()
