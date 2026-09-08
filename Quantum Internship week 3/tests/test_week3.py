"""Pytest suite for Week 3: kernel PSD/symmetry, VQE accuracy, assets, README math."""

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "day12_qsvm"))
sys.path.insert(0, str(ROOT / "day14_vqe_qaoa"))

EXPECTED_IMAGES = [
    "day11_loss.png",
    "day11_predictions.png",
    "day12_decision_boundaries.png",
    "day12_confusion_matrices.png",
    "day13_accuracy_vs_epochs.png",
    "day13_decision_regions.png",
    "day14_vqe_convergence.png",
    "day14_qaoa_ratio_vs_p.png",
    "day14_maxcut_graph.png",
    "day15_histograms.png",
    "day15_losses.png",
]


def test_kernel_symmetry_and_psd() -> None:
    """Quantum kernel Gram matrix is symmetric and positive semidefinite."""
    from task9_qsvm import kernel_matrix

    rng = np.random.RandomState(0)
    X = np.pi * rng.rand(8, 2)
    K = kernel_matrix(X, X)
    assert np.allclose(K, K.T, atol=1e-8)
    assert np.allclose(np.diag(K), 1.0, atol=1e-8)
    assert np.min(np.linalg.eigvalsh(K)) >= -1e-8


def test_vqe_within_tolerance() -> None:
    """VQE ground energy matches exact diagonalization within 1e-3 Ha."""
    from task11_vqe_qaoa import exact_energy, run_vqe

    vqe = run_vqe()
    exact, _ = exact_energy()
    assert abs(vqe["vqe"] - exact) < 1e-3


def test_images_exist() -> None:
    """Every committed figure exists and is non-empty."""
    for name in EXPECTED_IMAGES:
        path = ROOT / "assets" / name
        assert path.exists(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"


def test_readme_math_parity() -> None:
    """README $ and $$ delimiters have even parity; no raw table-pipe math."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count("$$") % 2 == 0
    assert text.replace("$$", "").count("$") % 2 == 0
    assert r"\ket" not in text and r"\bra" not in text
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            assert "$$" not in line
            for span in re.findall(r"\$[^$]*?\$", line):
                assert "|" not in span


def test_readme_image_links_resolve() -> None:
    """Every local README ](...) path exists on disk."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    paths = re.findall(r"\]\(([^)]+)\)", text)
    local = [p for p in paths if not p.startswith(("http://", "https://", "#"))]
    assert local, "no local links found"
    for p in local:
        assert (ROOT / p).exists(), f"missing linked file {p}"
