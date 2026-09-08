"""Day 12 — quantum kernel SVM vs classical RBF SVM.

Quantum kernel: ZZFeatureMap-style PennyLane embedding; kernel entries are
squared state overlaps. Fed to sklearn SVC(kernel='precomputed').
Datasets: moons-like toy + deterministic synthetic HEP-like
signal/background (fixed seed; used because no CERN Open Data download is
attempted in this offline build). All local.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SEED = 42
N_QUBITS = 2

ASSETS = Path(__file__).resolve().parent.parent / "assets"
_DEV = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_DEV, interface="numpy")
def _state_qnode(x: np.ndarray) -> np.ndarray:
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.RZ(2.0 * x[0], wires=0)
    qml.RZ(2.0 * x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(2.0 * (np.pi - x[0]) * (np.pi - x[1]), wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.state()


def feature_state(x: np.ndarray) -> np.ndarray:
    """Return the ZZ-style feature state for one 2D point."""
    return np.asarray(_state_qnode(np.asarray(x, dtype=float)))


def kernel_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared-overlap kernel between rows of A and rows of B."""
    sa = np.array([feature_state(x) for x in A])
    sb = np.array([feature_state(x) for x in B])
    return np.abs(sa.conj() @ sb.T) ** 2


def scale_to_angles(X: np.ndarray) -> np.ndarray:
    """Min-max scale features to [0, pi] for angle embedding."""
    lo, hi = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
    return np.pi * (X - lo) / np.maximum(hi - lo, 1e-12)


def make_hep_like(n: int = 120, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic HEP-like signal/background dataset."""
    rng = np.random.RandomState(seed)
    sig = rng.normal(loc=[1.0, 1.0], scale=0.35, size=(n // 2, 2))
    bkg = rng.normal(loc=[-0.2, -0.2], scale=0.9, size=(n - n // 2, 2))
    X = np.vstack([sig, bkg])
    y = np.array([1] * (n // 2) + [0] * (n - n // 2))
    perm = rng.permutation(n)
    return X[perm], y[perm]


def evaluate(
    X: np.ndarray, y: np.ndarray, name: str
) -> dict:
    """Train quantum-kernel and RBF SVMs; return accuracies and splits."""
    Xs = scale_to_angles(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.25, random_state=SEED, stratify=y
    )
    Ktr = kernel_matrix(Xtr, Xtr)
    Kte = kernel_matrix(Xte, Xtr)
    clf_q = SVC(kernel="precomputed")
    clf_q.fit(Ktr, ytr)
    acc_q = accuracy_score(yte, clf_q.predict(Kte))
    clf_r = SVC(kernel="rbf")
    clf_r.fit(Xtr, ytr)
    acc_r = accuracy_score(yte, clf_r.predict(Xte))
    return {
        "name": name,
        "Xtr": Xtr,
        "Xte": Xte,
        "ytr": ytr,
        "yte": yte,
        "clf_q": clf_q,
        "clf_r": clf_r,
        "acc_q": float(acc_q),
        "acc_r": float(acc_r),
        "Ktr": Ktr,
    }


def _boundary_ax(ax: plt.Axes, clf: SVC, Xtr: np.ndarray, ytr: np.ndarray,
                 Ktr: np.ndarray, title: str, precomputed: bool) -> None:
    h = 0.06
    x0 = np.arange(Xtr[:, 0].min() - 0.3, Xtr[:, 0].max() + 0.3, h)
    x1 = np.arange(Xtr[:, 1].min() - 0.3, Xtr[:, 1].max() + 0.3, h)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    if precomputed:
        Kg = kernel_matrix(grid, Xtr)
        zz = clf.predict(Kg).reshape(xx.shape)
    else:
        zz = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.3)
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, edgecolors="k", s=14)
    ax.set_title(title)


def save_figures(moons: dict, hep: dict) -> None:
    """Decision boundaries side-by-side + confusion matrices."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _boundary_ax(axes[0, 0], moons["clf_q"], moons["Xtr"], moons["ytr"],
                 moons["Ktr"], "Moons: quantum kernel", True)
    _boundary_ax(axes[0, 1], moons["clf_r"], moons["Xtr"], moons["ytr"],
                 moons["Ktr"], "Moons: RBF", False)
    _boundary_ax(axes[1, 0], hep["clf_q"], hep["Xtr"], hep["ytr"],
                 hep["Ktr"], "HEP-like: quantum kernel", True)
    _boundary_ax(axes[1, 1], hep["clf_r"], hep["Xtr"], hep["ytr"],
                 hep["Ktr"], "HEP-like: RBF", False)
    fig.suptitle("Day 12: decision boundaries")
    fig.savefig(ASSETS / "day12_decision_boundaries.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, res, key, title in [
        (axes[0, 0], moons, "q", "Moons quantum"),
        (axes[0, 1], moons, "r", "Moons RBF"),
        (axes[1, 0], hep, "q", "HEP quantum"),
        (axes[1, 1], hep, "r", "HEP RBF"),
    ]:
        clf = res["clf_q"] if key == "q" else res["clf_r"]
        if key == "q":
            pred = clf.predict(kernel_matrix(res["Xte"], res["Xtr"]))
        else:
            pred = clf.predict(res["Xte"])
        cm = confusion_matrix(res["yte"], pred)
        im = ax.imshow(cm)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Day 12: confusion matrices")
    fig.savefig(ASSETS / "day12_confusion_matrices.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run both comparisons, save figures, print accuracy table."""
    Xm, ym = make_moons(n_samples=120, noise=0.15, random_state=SEED)
    Xh, yh = make_hep_like()
    moons = evaluate(Xm, ym, "moons")
    hep = evaluate(Xh, yh, "hep")
    save_figures(moons, hep)
    print(f"moons_qsvm_acc={moons['acc_q']:.4f} moons_rbf_acc={moons['acc_r']:.4f}")
    print(f"hep_qsvm_acc={hep['acc_q']:.4f} hep_rbf_acc={hep['acc_r']:.4f}")


if __name__ == "__main__":
    main()
