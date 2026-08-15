#!/usr/bin/env python3
"""Reproducible comparison of a numerical 840-subconfiguration with the
canonical quaternionic 840-point kissing arrangement.

The script performs all requested steps:

(a) Construct the canonical quaternionic 840-point arrangement from the
    affine binary parity-lift description and save it as ``canonical840.txt``.
    The ordering is

        1--24    E1 = {(c,0,0): c in C},
        25--48   E2 = {(0,c,0): c in C},
        49--432  M1,
        433--816 M2,
        817--840 E3 = {(0,0,c): c in C}.

    Thus rows 817--840 are exactly the 24 vectors in the last R^4.

(b) Read a 841 x 12 numerical kissing arrangement, delete the special point
    (row 751 by default, using one-based indexing), and save the remaining
    rows as ``840_from_841.txt``.

(c) Approximately solve

        min_{pi in S_840, U in O(12)}
            sum_i ||x_i - U y_{pi(i)}||_2^2

    by multistart alternating Hungarian assignment and orthogonal Procrustes.
    This is a nonconvex problem; the output is the best solution found, not a
    proof of the global optimum.

(d) Draw the order-statistics plot of the 840 squared residuals.

(e) Test whether all 24 largest residuals are matched to rows 817--840 of the
    canonical arrangement, i.e. to E3 in the last R^4.

Coordinate convention
---------------------
Vectors are stored as rows.  The output permutation file contains pi(i) on
line i, with one-based indices.  The matrix U is stored in the usual column-
vector convention, so

    x_i approximately U y_{pi(i)}.

Internally this is evaluated as ``Y[pi] @ U.T``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


SQRT2 = np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Quaternionic construction
# ---------------------------------------------------------------------------

def qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product in coordinates (1,i,j,k)."""
    a, b, c, d = p
    e, f, g, h = q
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        dtype=np.float64,
    )


def qpower(q: np.ndarray, exponent: int) -> np.ndarray:
    """Nonnegative integer power of a quaternion."""
    result = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(exponent):
        result = qmul(result, q)
    return result


def sign_lifted_parity_fiber(
    eta: int,
    representatives: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the 128-element independent sign lift of

        B_eta = {(u,v,w) in (F_2^2)^3 : u+v+w=eta}.

    Elements of F_2^2 are encoded by 0,1,2,3 and addition is XOR.
    """
    output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for u in range(4):
        for v in range(4):
            w = eta ^ u ^ v
            for su, sv, sw in product((1.0, -1.0), repeat=3):
                output.append(
                    (
                        su * representatives[u],
                        sv * representatives[v],
                        sw * representatives[w],
                    )
                )
    if len(output) != 128:
        raise RuntimeError("A lifted parity fiber must contain 128 triples")
    return output


def construct_canonical_840() -> tuple[np.ndarray, list[str]]:
    """Construct the canonical quaternionic 840-point arrangement.

    This implements the formulas

        C = Q8 union omega Q8 union omega^2 Q8,
        D = tau C,

        T1_r = (tau omega^r, omega^r, omega^r)[Bhat_0],
        T2_r = (omega^r, tau omega^r, tau omega^r)[Bhat_delta_r],

    with delta_0=j, delta_1=i, delta_2=k, followed by the weighted
    embeddings defining M1 and M2.
    """
    one = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    qj = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    qk = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    representatives = [one, qi, qj, qk]

    omega = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    tau = np.array([1.0 / SQRT2, 1.0 / SQRT2, 0.0, 0.0], dtype=np.float64)
    omega_powers = [qpower(omega, r) for r in range(3)]

    # Fixed deterministic ordering of the Hurwitz 24-cell C.
    C = np.array(
        [
            qmul(omega_powers[r], sign * representatives[u])
            for r in range(3)
            for u in range(4)
            for sign in (1.0, -1.0)
        ],
        dtype=np.float64,
    )

    # delta_0=j, delta_1=i, delta_2=k in the encoding
    # 0=1, 1=i, 2=j, 3=k.
    deltas = [2, 1, 3]
    lifted_zero = sign_lifted_parity_fiber(0, representatives)
    lifted_shifted = [
        sign_lifted_parity_fiber(delta, representatives) for delta in deltas
    ]

    T1: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    T2: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for r in range(3):
        wr = omega_powers[r]
        twr = qmul(tau, wr)

        for a, b, c in lifted_zero:
            T1.append((qmul(twr, a), qmul(wr, b), qmul(wr, c)))

        for a, b, c in lifted_shifted[r]:
            T2.append((qmul(wr, a), qmul(twr, b), qmul(twr, c)))

    if len(T1) != 384 or len(T2) != 384:
        raise RuntimeError("Each canonical ternary relation must have size 384")

    rows: list[np.ndarray] = []
    labels: list[str] = []

    # E1 and E2.
    for c in C:
        rows.append(np.r_[c, np.zeros(8)])
        labels.append("E1")
    for c in C:
        rows.append(np.r_[np.zeros(4), c, np.zeros(4)])
        labels.append("E2")

    # M1 and M2.
    for a, b, c in T1:
        rows.append(np.r_[a / SQRT2, b / 2.0, c / 2.0])
        labels.append("M1")
    for a, b, c in T2:
        rows.append(np.r_[a / 2.0, b / SQRT2, c / 2.0])
        labels.append("M2")

    # E3 is deliberately last: rows 817--840.
    for c in C:
        rows.append(np.r_[np.zeros(8), c])
        labels.append("E3")

    Y = np.asarray(rows, dtype=np.float64)
    if Y.shape != (840, 12):
        raise RuntimeError(f"Unexpected canonical array shape: {Y.shape}")
    if np.unique(np.round(Y, decimals=14), axis=0).shape[0] != 840:
        raise RuntimeError("The canonical construction contains duplicate vectors")

    norms = np.linalg.norm(Y, axis=1)
    gram = Y @ Y.T
    np.fill_diagonal(gram, -np.inf)
    if np.max(np.abs(norms - 1.0)) > 1e-12:
        raise RuntimeError("Canonical unit-norm check failed")
    if np.max(gram) > 0.5 + 1e-12:
        raise RuntimeError("Canonical kissing-condition check failed")

    return Y, labels


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def normalize_rows(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("A zero row was encountered")
    return A / norms


def random_orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(R) >= 0.0, 1.0, -1.0)
    return Q @ np.diag(signs)


def nearest_orthogonal(A: np.ndarray, proper: bool = False) -> np.ndarray:
    left, _, right_t = np.linalg.svd(A, full_matrices=False)
    if proper and np.linalg.det(left @ right_t) < 0.0:
        left[:, -1] *= -1.0
    return left @ right_t


def procrustes(A: np.ndarray, B: np.ndarray, proper: bool = False) -> np.ndarray:
    """Return R minimizing ||A R-B||_F over O(d), or over SO(d)."""
    left, _, right_t = np.linalg.svd(A.T @ B, full_matrices=False)
    if proper and np.linalg.det(left @ right_t) < 0.0:
        left[:, -1] *= -1.0
    return left @ right_t


@dataclass
class CellCandidate:
    value: float
    rotation: np.ndarray
    permutation: np.ndarray


def find_cell_candidates(
    source_cell: np.ndarray,
    target_cell: np.ndarray,
    rng: np.random.Generator,
    restarts: int,
    keep: int,
    iterations: int,
) -> list[CellCandidate]:
    """Find symmetry-related alignments of one 24-point cell to another."""
    candidates: list[CellCandidate] = []

    for restart in range(restarts):
        rotation = (
            np.eye(source_cell.shape[1])
            if restart == 0
            else random_orthogonal(rng, source_cell.shape[1])
        )
        previous = np.inf
        permutation = np.arange(24)

        for _ in range(iterations):
            transformed = source_cell @ rotation
            cost = np.sum(
                (target_cell[:, None, :] - transformed[None, :, :]) ** 2,
                axis=2,
            )
            target_rows, source_rows = linear_sum_assignment(cost)
            # target_rows is sorted 0,...,23 for a square assignment.
            permutation = source_rows[np.argsort(target_rows)]
            rotation = procrustes(source_cell[permutation], target_cell)
            differences = source_cell[permutation] @ rotation - target_cell
            value = float(np.sum(differences * differences))
            if abs(previous - value) < 1e-14:
                break
            previous = value

        # Deduplicate symmetry-equivalent starts numerically.
        if all(
            np.linalg.norm(rotation - candidate.rotation) > 1e-6
            for candidate in candidates
        ):
            candidates.append(CellCandidate(value, rotation.copy(), permutation.copy()))

    candidates.sort(key=lambda candidate: candidate.value)
    if not candidates:
        raise RuntimeError("No 24-cell initialization was found")
    return candidates[:keep]


def alternating_assignment_procrustes(
    X: np.ndarray,
    Y: np.ndarray,
    row_map: np.ndarray,
    proper: bool,
    iterations: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """Alternate Hungarian assignment and orthogonal Procrustes."""
    previous = np.inf
    permutation = np.arange(X.shape[0])
    errors = np.full(X.shape[0], np.inf)

    for iteration in range(1, iterations + 1):
        transformed = Y @ row_map
        # Both arrays have unit rows, hence squared distance is 2-2<x,y>.
        cost = 2.0 - 2.0 * (X @ transformed.T)
        x_rows, y_rows = linear_sum_assignment(cost)
        permutation = y_rows[np.argsort(x_rows)]

        row_map = procrustes(Y[permutation], X, proper=proper)
        differences = X - Y[permutation] @ row_map
        errors = np.sum(differences * differences, axis=1)
        value = float(np.sum(errors))

        if abs(previous - value) < 1e-12:
            break
        previous = value

    return value, row_map, permutation, errors, iteration


def original_841_index(x840_zero_based: int, deleted_one_based: int) -> int:
    """Map a row of the reduced 840-array back to the original 841-array."""
    reduced_one_based = x840_zero_based + 1
    if reduced_one_based < deleted_one_based:
        return reduced_one_based
    return reduced_one_based + 1


def family_from_y_index(y_zero_based: int) -> str:
    if y_zero_based < 24:
        return "E1"
    if y_zero_based < 48:
        return "E2"
    if y_zero_based < 432:
        return "M1"
    if y_zero_based < 816:
        return "M2"
    return "E3"


def resolve_input_path(requested: str) -> Path:
    """Resolve a convenient default while remaining explicit and reproducible."""
    path = Path(requested)
    if path.exists():
        return path

    if requested == "gram841_coordinates.txt":
        fallbacks = [
            Path("gram841_coordinates(1).txt"),
            Path("gram841_coordinates (1).txt"),
        ]
        existing = [candidate for candidate in fallbacks if candidate.exists()]
        if len(existing) == 1:
            return existing[0]

    raise FileNotFoundError(
        f"Cannot find {requested!r}. Supply the path explicitly with --input."
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", default="gram841_coordinates.txt")
    parser.add_argument("--special-index", type=int, default=751,
                        help="One-based row deleted from the 841-point array")
    parser.add_argument("--canonical-output", default="canonical840.txt")
    parser.add_argument("--numerical-output", default="840_from_841.txt")
    parser.add_argument("--family-output", default="canonical840_families.txt")
    parser.add_argument("--out-prefix", default="canonical840_alignment")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--warm-start-U",
        default=None,
        help="Optional 12x12 orthogonal matrix in column-vector convention",
    )
    parser.add_argument(
        "--warm-start-permutation",
        default=None,
        help="Optional 840-line one-based permutation pi(i)",
    )

    # Search controls.  The defaults are intended for a serious reproducible run.
    parser.add_argument("--cell-restarts", type=int, default=1500)
    parser.add_argument("--cell-candidates", type=int, default=80)
    parser.add_argument("--cell-iterations", type=int, default=40)
    parser.add_argument("--global-trials", type=int, default=500)
    parser.add_argument("--global-iterations", type=int, default=40)
    parser.add_argument("--local-trials", type=int, default=500)
    parser.add_argument("--axial-edge-threshold", type=float, default=0.1)
    parser.add_argument(
        "--proper",
        action="store_true",
        help="Restrict U to SO(12); otherwise search over all O(12)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # (a) Construct and save the canonical arrangement.
    Y, labels = construct_canonical_840()
    np.savetxt(args.canonical_output, Y, fmt="%.17g")
    with open(args.family_output, "w", encoding="utf-8") as handle:
        handle.write("# y_index family\n")
        for index, label in enumerate(labels, start=1):
            handle.write(f"{index} {label}\n")

    # (b) Read the 841-point arrangement and delete the special row.
    input_path = resolve_input_path(args.input)
    X841 = np.loadtxt(input_path, dtype=np.float64)
    if X841.shape != (841, 12):
        raise ValueError(f"Expected an 841x12 array, got {X841.shape}")
    if not 1 <= args.special_index <= 841:
        raise ValueError("--special-index must lie between 1 and 841")

    X841 = normalize_rows(X841)
    X = np.delete(X841, args.special_index - 1, axis=0)
    X = normalize_rows(X)
    np.savetxt(args.numerical_output, X, fmt="%.17g")

    x_gram = X @ X.T
    np.fill_diagonal(x_gram, -np.inf)
    y_gram = Y @ Y.T
    np.fill_diagonal(y_gram, -np.inf)

    print(f"Read numerical input: {input_path}")
    print(f"Deleted original row: {args.special_index}")
    print(f"Saved numerical 840-set: {args.numerical_output}")
    print(f"Saved canonical 840-set: {args.canonical_output}")
    print(f"Numerical max inner product: {np.max(x_gram):.17g}")
    print(f"Canonical max inner product: {np.max(y_gram):.17g}")

    # Structural 8+4 decomposition.  The canonical frame operator has
    # eigenvalues 78 (multiplicity 8) and 54 (multiplicity 4).  For X, the
    # bottom four eigendirections identify the deformed final R^4 factor.
    frame_eigenvalues, frame_eigenvectors = np.linalg.eigh(X.T @ X)
    low_basis = frame_eigenvectors[:, :4]
    low_energies = np.sum((X @ low_basis) ** 2, axis=1)

    # The 24 largest low-space energies are the numerical special 24-set.
    special24_indices = np.argsort(low_energies)[-24:]
    special24_target = X[special24_indices] @ low_basis

    # The 48 smallest low-space energies form E1 and E2.  Split them into two
    # connected 24-point cells using their within-set Gram matrix.
    axial48_indices = np.argsort(low_energies)[:48]
    axial_gram = X[axial48_indices] @ X[axial48_indices].T
    adjacency = np.abs(axial_gram) > args.axial_edge_threshold
    np.fill_diagonal(adjacency, False)
    component_count, component_labels = connected_components(
        csr_matrix(adjacency), directed=False
    )
    component_sizes = np.bincount(component_labels)
    if component_count != 2 or sorted(component_sizes.tolist()) != [24, 24]:
        raise RuntimeError(
            "The 48 axial candidates did not split into two 24-point "
            f"components; sizes={component_sizes.tolist()}"
        )

    # Exact Hurwitz 24-cell, read from E3 in the canonical ordering.
    C = Y[816:840, 8:12]

    axial_data: list[tuple[np.ndarray, np.ndarray, list[CellCandidate]]] = []
    for component in range(2):
        indices = axial48_indices[component_labels == component]
        _, _, right_t = np.linalg.svd(X[indices], full_matrices=False)
        basis = right_t[:4].T
        target = X[indices] @ basis
        candidates = find_cell_candidates(
            C,
            target,
            rng,
            restarts=args.cell_restarts,
            keep=args.cell_candidates,
            iterations=args.cell_iterations,
        )
        axial_data.append((indices, basis, candidates))
        print(
            f"Axial component {component + 1}: best 24-cell error "
            f"{candidates[0].value:.12g}; candidates={len(candidates)}"
        )

    # Use the deformed special 24-set to initialize the last R^4 orientation.
    special_candidates = find_cell_candidates(
        C,
        special24_target,
        rng,
        restarts=args.cell_restarts,
        keep=args.cell_candidates,
        iterations=args.cell_iterations,
    )
    print(
        "Special final-R4 component: best 24-cell error "
        f"{special_candidates[0].value:.12g}; "
        f"candidates={len(special_candidates)}"
    )

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None

    # An optional published/reproducibility seed can be checked and refined
    # before the independent multistart search.  The seed does not change the
    # objective or the subsequent optimization.
    if (args.warm_start_U is None) != (args.warm_start_permutation is None):
        raise ValueError(
            "Supply both --warm-start-U and --warm-start-permutation, or neither"
        )
    if args.warm_start_U is not None:
        warm_U = np.loadtxt(args.warm_start_U, dtype=np.float64)
        warm_permutation = np.loadtxt(
            args.warm_start_permutation, dtype=np.int64
        ).reshape(-1) - 1
        if warm_U.shape != (12, 12):
            raise ValueError("The warm-start U must have shape 12x12")
        if warm_permutation.shape != (840,) or set(warm_permutation.tolist()) != set(range(840)):
            raise ValueError("The warm-start permutation must be a permutation of 1,...,840")
        warm_row_map = nearest_orthogonal(warm_U.T, proper=args.proper)
        warm_value, warm_row_map, warm_permutation, warm_errors, _ = (
            alternating_assignment_procrustes(
                X,
                Y,
                warm_row_map,
                proper=args.proper,
                iterations=args.global_iterations,
            )
        )
        best = (
            warm_value,
            warm_row_map.copy(),
            warm_permutation.copy(),
            warm_errors.copy(),
        )
        print(
            f"Warm start after assignment/Procrustes refinement: "
            f"total={warm_value:.12g}, RMS={np.sqrt(warm_value / 840):.9g}"
        )

    # (c) Multistart global assignment/Procrustes search.  Try both exchanges
    # of the two high-frame axial cells E1 and E2.
    for trial in range(1, args.global_trials + 1):
        swap = bool(trial % 2)
        first = axial_data[1] if swap else axial_data[0]
        second = axial_data[0] if swap else axial_data[1]

        candidate1 = first[2][rng.integers(len(first[2]))]
        candidate2 = second[2][rng.integers(len(second[2]))]
        candidate3 = special_candidates[rng.integers(len(special_candidates))]

        row_map = np.zeros((12, 12), dtype=np.float64)
        row_map[0:4, :] = candidate1.rotation @ first[1].T
        row_map[4:8, :] = candidate2.rotation @ second[1].T
        row_map[8:12, :] = candidate3.rotation @ low_basis.T
        row_map = nearest_orthogonal(row_map, proper=args.proper)

        value, row_map, permutation, errors, _ = alternating_assignment_procrustes(
            X,
            Y,
            row_map,
            proper=args.proper,
            iterations=args.global_iterations,
        )

        if best is None or value < best[0]:
            best = (value, row_map.copy(), permutation.copy(), errors.copy())
            top24 = np.argsort(errors)[-24:]
            e3_count = int(np.sum(permutation[top24] >= 816))
            print(
                f"New best, global trial {trial}: total={value:.12g}, "
                f"RMS={np.sqrt(value / 840):.9g}, "
                f"E3 in largest 24={e3_count}/24"
            )

    if best is None:
        raise RuntimeError(
            "The search produced no result. Use positive --global-trials or "
            "supply a warm start."
        )

    # Local perturbation search around the current best orthogonal map.
    scales = (0.01, 0.02, 0.05, 0.10, 0.20)
    for local_trial in range(args.local_trials):
        scale = scales[local_trial % len(scales)]
        start = nearest_orthogonal(
            best[1] + scale * rng.normal(size=(12, 12)),
            proper=args.proper,
        )
        value, row_map, permutation, errors, _ = alternating_assignment_procrustes(
            X,
            Y,
            start,
            proper=args.proper,
            iterations=args.global_iterations,
        )
        if value < best[0]:
            best = (value, row_map.copy(), permutation.copy(), errors.copy())
            print(
                f"Improved locally: total={value:.12g}, "
                f"RMS={np.sqrt(value / 840):.9g}"
            )

    total, row_map, permutation, errors = best
    U = row_map.T  # x approximately U y in column-vector convention.
    aligned_Y = Y[permutation] @ U.T

    prefix = Path(args.out_prefix)
    np.savetxt(f"{prefix}_U.txt", U, fmt="%.17g")
    np.savetxt(f"{prefix}_permutation.txt", permutation[:, None] + 1, fmt="%d")
    np.savetxt(f"{prefix}_aligned_y.txt", aligned_Y, fmt="%.17g")
    np.savetxt(f"{prefix}_order_statistics.txt", np.sort(errors), fmt="%.17g")

    order = np.argsort(errors)
    ranks = np.empty(840, dtype=np.int64)
    ranks[order] = np.arange(1, 841)

    with open(f"{prefix}_residuals.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "# x_index original_841_index pi_of_i y_family "
            "squared_error euclidean_error order_rank\n"
        )
        for i in range(840):
            handle.write(
                f"{i + 1} "
                f"{original_841_index(i, args.special_index)} "
                f"{permutation[i] + 1} "
                f"{family_from_y_index(int(permutation[i]))} "
                f"{errors[i]:.17g} "
                f"{np.sqrt(errors[i]):.17g} "
                f"{ranks[i]}\n"
            )

    # (d) Order-statistics plot.
    import matplotlib.pyplot as plt

    sorted_errors = np.sort(errors)
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(np.arange(1, 841), sorted_errors, linewidth=1.5)
    plt.axvline(816.5, linestyle="--", linewidth=1.0)
    plt.xlabel("Order-statistic index $i$")
    plt.ylabel(r"$i$-th ordered value of $\|x-Uy\|_2^2$")
    plt.title("Canonical quaternionic 840-point alignment")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{prefix}_order_statistics.png", dpi=240)
    plt.savefig(f"{prefix}_order_statistics.pdf")
    plt.close()

    # (e) Analyze the 24 largest residuals.
    top24 = order[-24:]
    top24_y = permutation[top24]
    top24_e3_mask = top24_y >= 816
    e3_count = int(np.sum(top24_e3_mask))
    all_top24_are_e3 = bool(e3_count == 24)
    detected_special_set_agrees = bool(
        set(top24.tolist()) == set(special24_indices.tolist())
    )

    core_mask = permutation < 816
    e3_mask = permutation >= 816

    with open(f"{prefix}_top24.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "# rank x_index original_841_index pi_of_i y_family "
            "squared_error euclidean_error\n"
        )
        for i in top24:
            handle.write(
                f"{ranks[i]} {i + 1} "
                f"{original_841_index(i, args.special_index)} "
                f"{permutation[i] + 1} "
                f"{family_from_y_index(int(permutation[i]))} "
                f"{errors[i]:.17g} {np.sqrt(errors[i]):.17g}\n"
            )

    with open(f"{prefix}_report.txt", "w", encoding="utf-8") as handle:
        handle.write("CANONICAL QUATERNIONIC 840 ALIGNMENT\n")
        handle.write("===================================\n\n")
        handle.write(
            "The optimization alternates exact Hungarian assignments with "
            "orthogonal Procrustes updates from multiple structural starts.\n"
        )
        handle.write(
            "Because the joint permutation/orthogonal problem is nonconvex, "
            "the value below is the best value found, not a proof of the "
            "global minimum.\n\n"
        )
        handle.write(f"Input file: {input_path}\n")
        handle.write(f"Deleted original one-based row: {args.special_index}\n")
        handle.write(f"Canonical coordinate file: {args.canonical_output}\n")
        handle.write(f"Numerical coordinate file: {args.numerical_output}\n\n")
        handle.write(f"Total squared error: {total:.17g}\n")
        handle.write(f"Mean squared error: {np.mean(errors):.17g}\n")
        handle.write(f"RMS Euclidean error: {np.sqrt(np.mean(errors)):.17g}\n")
        handle.write(f"Maximum squared error: {np.max(errors):.17g}\n")
        handle.write(f"det(U): {np.linalg.det(U):.17g}\n")
        handle.write(
            "Maximum orthogonality error: "
            f"{np.max(np.abs(U.T @ U - np.eye(12))):.3e}\n\n"
        )
        handle.write("Canonical rows 1--816 (E1,E2,M1,M2):\n")
        handle.write(f"  total squared error: {np.sum(errors[core_mask]):.17g}\n")
        handle.write(
            f"  RMS Euclidean error: {np.sqrt(np.mean(errors[core_mask])):.17g}\n"
        )
        handle.write(
            f"  maximum squared error: {np.max(errors[core_mask]):.17g}\n\n"
        )
        handle.write("Canonical rows 817--840 (E3 in the last R^4):\n")
        handle.write(f"  total squared error: {np.sum(errors[e3_mask]):.17g}\n")
        handle.write(
            f"  RMS Euclidean error: {np.sqrt(np.mean(errors[e3_mask])):.17g}\n"
        )
        handle.write(f"  minimum squared error: {np.min(errors[e3_mask]):.17g}\n")
        handle.write(f"  maximum squared error: {np.max(errors[e3_mask]):.17g}\n\n")
        handle.write(
            f"E3 vectors among the 24 largest residuals: {e3_count}/24\n"
        )
        handle.write(
            "Do all 24 largest residuals belong to E3? "
            f"{all_top24_are_e3}\n"
        )
        handle.write(
            "Do the 24 largest-residual numerical rows equal the 24 rows "
            "detected independently by largest last-R4 projection energy? "
            f"{detected_special_set_agrees}\n\n"
        )
        handle.write(
            f"Order-statistic boundary e_(816): {sorted_errors[815]:.17g}\n"
        )
        handle.write(
            f"Order-statistic boundary e_(817): {sorted_errors[816]:.17g}\n"
        )
        handle.write(
            f"Largest order statistic e_(840): {sorted_errors[-1]:.17g}\n"
        )

    print("\nFinal result")
    print(f"  Best total squared error found: {total:.15g}")
    print(f"  RMS Euclidean error: {np.sqrt(np.mean(errors)):.15g}")
    print(f"  det(U): {np.linalg.det(U):.15g}")
    print(f"  E3 among the 24 largest residuals: {e3_count}/24")
    print(f"  All 24 largest residuals are E3: {all_top24_are_e3}")
    print(f"  Report: {prefix}_report.txt")
    print(f"  Plot: {prefix}_order_statistics.png")


if __name__ == "__main__":
    main()
