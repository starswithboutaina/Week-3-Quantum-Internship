"""Day 14 — VQE on a 2-qubit H2-like Hamiltonian + QAOA MaxCut.

VQE uses parameter-shift gradients and is cross-checked against exact numpy
diagonalization and blackhole.expectation when importable. QAOA (p=1,2) runs
on a 6-node MaxCut graph. All local, fixed seeds.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

SEED = 42
VQE_STEPS = 250
VQE_LR = 0.10
N_QAOA = 6
EDGES: tuple = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4))

# Explicit Pauli coefficients of the H2-like Hamiltonian.
COEFFS = {"II": -0.4804, "ZI": 0.3935, "IZ": 0.3935, "ZZ": -0.0113, "XX": 0.1812}

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def get_device(wires: int) -> qml.devices.Device:
    """Local simulator, Lightning preferred."""
    try:
        return qml.device("lightning.qubit", wires=wires)
    except Exception:
        return qml.device("default.qubit", wires=wires)


def hamiltonian() -> qml.Hamiltonian:
    """Build the H2-like Hamiltonian from explicit coefficients."""
    I, X, Z = qml.Identity(0), qml.PauliX(0), qml.PauliZ(0)
    H = (
        COEFFS["II"] * I
        + COEFFS["ZI"] * qml.PauliZ(0)
        + COEFFS["IZ"] * qml.PauliZ(1)
        + COEFFS["ZZ"] * (qml.PauliZ(0) @ qml.PauliZ(1))
        + COEFFS["XX"] * (qml.PauliX(0) @ qml.PauliX(1))
    )
    return H


def exact_energy() -> tuple[float, np.ndarray]:
    """Exact ground energy via numpy diagonalization of the 4x4 matrix."""
    I = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    H = (
        COEFFS["II"] * np.eye(4)
        + COEFFS["ZI"] * np.kron(Z, I)
        + COEFFS["IZ"] * np.kron(I, Z)
        + COEFFS["ZZ"] * np.kron(Z, Z)
        + COEFFS["XX"] * np.kron(X, X)
    )
    vals, vecs = np.linalg.eigh(H)
    return float(vals[0]), H


def blackhole_check(H: np.ndarray, psi: np.ndarray) -> float | None:
    """Cross-check <psi|H|psi> with blackhole.expectation if importable."""
    try:
        import blackhole

        return float(blackhole.expectation(np.asarray(psi), np.asarray(H)))
    except Exception:
        return None


def run_vqe(seed: int = SEED, steps: int = VQE_STEPS) -> dict:
    """VQE with parameter-shift gradients; return energies and state."""
    np.random.seed(seed)
    dev = get_device(2)
    H = hamiltonian()

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def cost(weights: np.ndarray) -> float:
        qml.StronglyEntanglingLayers(weights, wires=range(2))
        return qml.expval(H)

    opt = qml.AdamOptimizer(stepsize=VQE_LR)
    weights = pnp.array(np.random.normal(0.0, 0.3, size=(2, 2, 3)),
                        requires_grad=True)
    energies: list[float] = []
    for _ in range(steps):
        weights, energy = opt.step_and_cost(cost, weights)
        energies.append(float(energy))
    exact, Hmat = exact_energy()

    @qml.qnode(get_device(2), interface="numpy")
    def state_qnode(w: np.ndarray) -> np.ndarray:
        qml.StronglyEntanglingLayers(w, wires=range(2))
        return qml.state()

    psi = np.asarray(state_qnode(weights))
    bh = blackhole_check(Hmat, psi)
    return {"energies": energies, "vqe": energies[-1], "exact": exact,
            "blackhole": bh, "weights": weights}


def maxcut_optimal() -> tuple[int, str]:
    """Brute-force optimal MaxCut value and bitstring for the 6-node graph."""
    best, bit = -1, "0" * N_QAOA
    for k in range(2 ** N_QAOA):
        s = format(k, f"0{N_QAOA}b")
        val = sum(1 for (i, j) in EDGES if s[i] != s[j])
        if val > best:
            best, bit = val, s
    return best, bit


def run_qaoa(p: int, seed: int = SEED, steps: int = 250) -> dict:
    """QAOA with p layers; return expected cut value and best bitstring."""
    np.random.seed(seed + p)
    dev = get_device(N_QAOA)
    coeffs = [0.5] * len(EDGES)
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) for (i, j) in EDGES]
    H_cost = 0.5 * len(EDGES) * qml.Identity(0) - 0.5 * qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def cost(params: pnp.ndarray) -> float:
        gammas, betas = params[:p], params[p:]
        for w in range(N_QAOA):
            qml.Hadamard(wires=w)
        for layer in range(p):
            for (i, j) in EDGES:
                qml.MultiRZ(2.0 * gammas[layer], wires=[i, j])
            for w in range(N_QAOA):
                qml.RX(2.0 * betas[layer], wires=w)
        return qml.expval(H_cost)

    opt = qml.AdamOptimizer(stepsize=0.10)
    init = np.concatenate([np.random.uniform(0.0, np.pi, size=p),
                           np.random.uniform(0.0, np.pi / 2, size=p)])
    params = pnp.array(init, requires_grad=True)

    @qml.qnode(get_device(N_QAOA), interface="numpy")
    def prob_qnode(params_: np.ndarray) -> np.ndarray:
        gammas_, betas_ = params_[:p], params_[p:]
        for w in range(N_QAOA):
            qml.Hadamard(wires=w)
        for layer in range(p):
            for (i, j) in EDGES:
                qml.MultiRZ(2.0 * gammas_[layer], wires=[i, j])
            for w in range(N_QAOA):
                qml.RX(2.0 * betas_[layer], wires=w)
        return qml.probs(wires=range(N_QAOA))

    best_val = -1.0
    for _ in range(steps):
        params, val = opt.step_and_cost(cost, params)
        best_val = max(best_val, float(val))
    probs = np.asarray(prob_qnode(params))
    rng = np.random.RandomState(seed + 100 * p)
    draws = rng.choice(2 ** N_QAOA, size=512, p=probs / probs.sum())
    best_bit, best_cut = format(int(np.argmax(probs)), f"0{N_QAOA}b"), -1
    for d in draws:
        bit = format(int(d), f"0{N_QAOA}b")
        cut = sum(1 for (i, j) in EDGES if bit[i] != bit[j])
        if cut > best_cut:
            best_bit, best_cut = bit, cut
    return {"expected": best_val, "bitstring": best_bit, "cut": best_cut}


def save_figures(vqe: dict, qaoa: dict, optimal: int) -> None:
    """VQE convergence, QAOA ratio-vs-p, and cut graph figures."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(vqe["energies"], label="VQE")
    ax.axhline(vqe["exact"], ls="--", label="Exact")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("Day 14: VQE convergence")
    ax.legend()
    fig.savefig(ASSETS / "day14_vqe_convergence.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ps = sorted(qaoa)
    ratios = [qaoa[p]["expected"] / optimal for p in ps]
    ax.bar([f"p={p}" for p in ps], ratios)
    ax.axhline(1.0, ls="--", label="Optimal")
    ax.set_ylabel("Approximation ratio")
    ax.set_title("Day 14: QAOA approximation ratio vs p")
    ax.legend()
    fig.savefig(ASSETS / "day14_qaoa_ratio_vs_p.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    best_p = max(ps, key=lambda p: qaoa[p]["expected"])
    bit = qaoa[best_p]["bitstring"]
    angles = np.linspace(0, 2 * np.pi, N_QAOA, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    colors = ["tab:blue" if b == "0" else "tab:orange" for b in bit]
    fig, ax = plt.subplots()
    for (i, j) in EDGES:
        cut = bit[i] != bit[j]
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], "r-" if cut else "k:",
                lw=2.5 if cut else 1.0,
                label="Cut edge" if cut else "Uncut")
    for i in range(N_QAOA):
        x, y = pos[i]
        ax.scatter([x], [y], c=[colors[i]], s=300, zorder=3, edgecolors="k")
        ax.text(x, y, str(i), ha="center", va="center", zorder=4)
    handles = [plt.Line2D([0], [0], color="r", lw=2.5, label="Cut edge"),
               plt.Line2D([0], [0], color="k", ls=":", label="Uncut")]
    ax.legend(handles=handles)
    ax.set_aspect("equal")
    ax.set_title(f"Day 14: MaxCut graph (p={best_p}, cut={qaoa[best_p]['cut']})")
    fig.savefig(ASSETS / "day14_maxcut_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run VQE + QAOA, save figures, print energies and ratios."""
    vqe = run_vqe()
    optimal, _ = maxcut_optimal()
    qaoa = {p: run_qaoa(p) for p in (1, 2)}
    save_figures(vqe, qaoa, optimal)
    print(f"vqe_energy={vqe['vqe']:.6f} exact_energy={vqe['exact']:.6f}")
    if vqe["blackhole"] is not None:
        print(f"blackhole_energy={vqe['blackhole']:.6f}")
    for p in (1, 2):
        print(f"qaoa_p{p}_ratio={qaoa[p]['expected'] / optimal:.4f} "
              f"cut={qaoa[p]['cut']}/{optimal}")


if __name__ == "__main__":
    main()
