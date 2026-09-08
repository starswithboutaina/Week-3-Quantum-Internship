"""Day 11 — hybrid classical-quantum regression pipeline.

Target: y = sin(x) * exp(-x / 5) on a fixed-seed grid.
Model: classical Linear -> PennyLane QNode (AngleEmbedding +
StronglyEntanglingLayers) wrapped as torch.nn.Module -> Linear.
Device: lightning.qubit with default.qubit fallback. All local.
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
N_QUBITS = 4
N_LAYERS = 2
N_POINTS = 80
EPOCHS = 120
LR = 0.05

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def get_device() -> qml.devices.Device:
    """Return a local simulator, preferring Lightning."""
    try:
        return qml.device("lightning.qubit", wires=N_QUBITS)
    except Exception:
        return qml.device("default.qubit", wires=N_QUBITS)


def make_dataset(n: int = N_POINTS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed synthetic regression target."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) * np.exp(-x / 5.0)
    jitter = rng.normal(0.0, 1e-6, size=n)  # keeps seed meaningful, target intact
    return x.astype(np.float32), (y + jitter).astype(np.float32)


def build_qnode(dev: qml.devices.Device) -> qml.QNode:
    """AngleEmbedding + entangler QNode returning one expval per wire."""

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    return circuit


class HybridRegressor(nn.Module):
    """MLP with a quantum layer in the middle."""

    def __init__(self, qnode: qml.QNode) -> None:
        super().__init__()
        self.pre = nn.Linear(1, N_QUBITS)
        self.qlayer = qml.qnn.TorchLayer(
            qnode, {"weights": (N_LAYERS, N_QUBITS, 3)}
        )
        self.post = nn.Linear(N_QUBITS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.pre(x)) * np.pi / 2.0
        h = self.qlayer(h)
        return self.post(h).squeeze(-1)


def train(seed: int = SEED) -> dict:
    """Train the hybrid model; return metrics and histories."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = get_device()
    qnode = build_qnode(dev)
    model = HybridRegressor(qnode)
    x_np, y_np = make_dataset()
    xb = torch.from_numpy(x_np).unsqueeze(-1)
    yb = torch.from_numpy(y_np)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        init_pred = model(xb)
        init_loss = float(loss_fn(init_pred, yb))
    losses: list[float] = []
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        final_pred = model(xb).numpy()
        final_loss = float(loss_fn(model(xb), yb))
    reduction = 100.0 * (init_loss - final_loss) / init_loss
    return {
        "x": x_np,
        "y": y_np,
        "pred": final_pred,
        "losses": losses,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "reduction_pct": reduction,
    }


def save_figures(res: dict) -> None:
    """Write loss curve and prediction-vs-target PNGs."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(res["losses"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Day 11: hybrid training loss")
    fig.savefig(ASSETS / "day11_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(res["x"], res["y"], "o", ms=3, label="Target")
    ax.plot(res["x"], res["pred"], "-", label="Hybrid prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Day 11: prediction vs target")
    ax.legend()
    fig.savefig(ASSETS / "day11_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run training, save figures, print metrics."""
    res = train()
    save_figures(res)
    print(f"final_mse={res['final_loss']:.6f}")
    print(f"loss_reduction_pct={res['reduction_pct']:.2f}")


if __name__ == "__main__":
    main()
