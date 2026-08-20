#!/usr/bin/env python3
"""
Fast existence checker for quaternionic constructions in R^(4k).

Input
-----
A whitespace-delimited N x d coordinate file (or .npy), with d divisible by 4.
Rows are normalized automatically.

Mathematical search
-------------------
We seek k=d/4 mutually orthogonal 4-spaces H_1,...,H_k such that

  * X contains a regular 24-cell S_j spanning H_j;
  * the projection of X to H_j satisfies the requested shell rule O.

A k-clique of admissible 4-spaces gives an orthogonal decomposition of R^(4k).
The script then rechecks every block on the full X and writes a certificate.

Speed ideas
-----------
1. Search 24-cells through antipodal LINES, not 24-tuples of points.
   A regular 24-cell has 12 antipodal lines.  If f1,...,f4 are an
   orthonormal frame from the cell, then its other 16 vertices are exactly

       (s1 f1 + s2 f2 + s3 f3 + s4 f4)/2,  s_i in {+1,-1}.

   Thus a candidate is certified by hash lookups, without projecting all X.

2. In --mode auto, use the lowest-support antipodal lines as the possible
   frame lines.  This is designed for canonical Barnes--Wall / Cohn--Li /
   Leech / R^28 coordinate files.  It avoids the enormous graph of all
   24-cells.  A positive result is rigorous; a negative auto result is only
   NOT FOUND.  --mode exhaustive uses all antipodal lines and is complete
   for the 24-cell enumeration, but may be expensive.

3. Apply the projection/shell test before a cell enters the graph.  With the
   built-in strict2o rule, a small deterministic sample is a safe rejection
   test: one bad projected direction already rejects the cell.

4. Build the orthogonality graph ONLINE.  As soon as an admissible cell is
   found, connect it to previous admissible cells using bit masks and ask
   whether it completes a k-clique.  We never materialize a dense all-cell
   graph and stop at the first certificate.

Shell rule O
------------
The exact family O is isolated behind a tiny interface.

Built-in rule:
  --O-rule strict2o
      after the selected 24-cell is identified with the Hurwitz 24-cell,
      every nonzero projected direction must lie in the binary octahedral
      group 2O.  Grouping by radii then gives
          Proj_H X = union_w w R_w,  R_w subseteq 2O.

Custom rule:
  --O-module my_O.py
where my_O.py defines

    def shell_in_O(R, radius, tol):
        # R is the array of DISTINCT unit directions on this nonzero shell.
        return True or False

Optionally it may also define the safe pointwise prefilter

    def pointwise_direction_ok(U, tol):
        # U is an M x 4 array of unit directions.
        # Return a Boolean array (or one bool).  It must NEVER reject a
        # direction that can occur in an allowed shell.

This lets the graph/search code stay unchanged when O is enlarged.

Examples
--------
  python check_quaternionic_construction.py coord.txt

  python check_quaternionic_construction.py CohnLi20.txt --mode auto \
      --O-module my_O.py --prefix r20

  python check_quaternionic_construction.py rotated_small.txt --mode exhaustive

Outputs on success (PREFIX defaults to input stem + '_quat'):
  PREFIX_certificate.json
  PREFIX_transform.txt
and, with --save-rotated,
  PREFIX_rotated.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# 4-dimensional standard objects
# -----------------------------------------------------------------------------

def hurwitz_24cell() -> np.ndarray:
    """24 Hurwitz units: ±e_i and all half-sign vectors."""
    out = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4)
            v[i] = s
            out.append(v)
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        out.append(np.asarray(signs, dtype=float) / 2.0)
    return np.asarray(out, dtype=float)


def d4_24cell() -> np.ndarray:
    """The other standard 24-cell: (±e_i±e_j)/sqrt(2)."""
    out = []
    a = 1.0 / math.sqrt(2.0)
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(4)
                    v[i] = si * a
                    v[j] = sj * a
                    out.append(v)
    return np.asarray(out, dtype=float)


HURWITZ = hurwitz_24cell()
TWO_O = np.vstack([HURWITZ, d4_24cell()])
# Numerical deduplication (the two 24-cells are disjoint in this convention).
TWO_O = np.unique(np.round(TWO_O, 14), axis=0)
assert HURWITZ.shape == (24, 4)
assert TWO_O.shape == (48, 4)


# -----------------------------------------------------------------------------
# I/O and quantized antipodal-line hashing
# -----------------------------------------------------------------------------

def load_coordinates(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        X = np.load(p)
    else:
        X = np.loadtxt(p, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("Input must be a 2D coordinate matrix.")
    n, d = X.shape
    if d % 4:
        raise ValueError(f"Dimension {d} is not divisible by 4.")
    nr = np.linalg.norm(X, axis=1)
    if np.any(nr <= 1e-14):
        raise ValueError("Zero row found in input.")
    X = np.asarray(X / nr[:, None], dtype=np.float64)
    return X


def canonicalize_lines(A: np.ndarray, zero_tol: float) -> np.ndarray:
    """Choose one sign for every nonzero row, invariant under x -> -x."""
    A = np.asarray(A, dtype=float).copy()
    nz = np.abs(A) > zero_tol
    if np.any(~np.any(nz, axis=1)):
        raise ValueError("Zero vector in canonicalize_lines.")
    first = np.argmax(nz, axis=1)
    s = np.sign(A[np.arange(len(A)), first])
    s[s == 0] = 1.0
    A *= s[:, None]
    A[np.abs(A) <= zero_tol] = 0.0
    return A


def quantize(A: np.ndarray, decimals: int) -> np.ndarray:
    scale = float(10 ** decimals)
    q = np.rint(np.asarray(A) * scale)
    # int64 gives abundant headroom even at decimals=12.
    return np.asarray(q, dtype=np.int64)


def row_bytes(Aint: np.ndarray) -> List[bytes]:
    Aint = np.ascontiguousarray(Aint)
    return [row.tobytes() for row in Aint]


@dataclass
class LineData:
    lines: np.ndarray                 # L x d, one unit representative per antipodal line
    source_rows: np.ndarray           # representative row in X
    multiplicities: np.ndarray        # number of rows quantizing to that projective line
    supports: np.ndarray
    keyset: set
    key_to_id: dict
    decimals: int
    zero_tol: float


def extract_antipodal_lines(X: np.ndarray,
                             decimals: int,
                             zero_tol: float,
                             require_both_signs: bool = True) -> LineData:
    """
    Extract projective lines by quantized canonical representatives.

    For a true antipodal spherical code, a line normally has multiplicity 2.
    With require_both_signs=True we retain only multiplicity >= 2, which is
    exactly what an axial regular 24-cell needs.
    """
    C = canonicalize_lines(X, zero_tol)
    Q = quantize(C, decimals)
    uq, first, counts = np.unique(Q, axis=0, return_index=True, return_counts=True)
    keep = counts >= 2 if require_both_signs else np.ones(len(counts), dtype=bool)
    uq = uq[keep]
    first = first[keep]
    counts = counts[keep]
    L = C[first]
    L /= np.linalg.norm(L, axis=1)[:, None]
    supports = np.sum(np.abs(L) > zero_tol, axis=1).astype(np.int16)
    key_list = row_bytes(uq)
    keys = set(key_list)
    key_to_id = {key: i for i, key in enumerate(key_list)}
    return LineData(L, first, counts, supports, keys, key_to_id, decimals, zero_tol)


def line_key(v: np.ndarray, decimals: int, zero_tol: float) -> bytes:
    c = canonicalize_lines(np.asarray(v, float)[None, :], zero_tol)[0]
    return quantize(c[None, :], decimals)[0].tobytes()


# -----------------------------------------------------------------------------
# 24-cell certification from an orthonormal frame
# -----------------------------------------------------------------------------

def frame_is_orthonormal(F: np.ndarray, tol: float) -> bool:
    return bool(np.max(np.abs(F @ F.T - np.eye(4))) <= tol)


def expected_24cell_lines(F: np.ndarray) -> np.ndarray:
    """
    Return the 12 projective lines of the Hurwitz 24-cell in frame F.

    Four are the frame axes.  The 16 half-sign vertices give 8 projective
    lines; fixing the first sign to +1 chooses one from each antipodal pair.
    """
    out = [F[i].copy() for i in range(4)]
    for s2, s3, s4 in itertools.product((-1.0, 1.0), repeat=3):
        coeff = np.asarray([1.0, s2, s3, s4]) / 2.0
        out.append(coeff @ F)
    return np.asarray(out)


def certify_24cell_frame(F: np.ndarray,
                          line_data: LineData,
                          orth_tol: float) -> Optional[Tuple[int, ...]]:
    """Return the 12 antipodal-line IDs of the cell, or None."""
    if not frame_is_orthonormal(F, orth_tol):
        return None
    ids = []
    for v in expected_24cell_lines(F):
        key = line_key(v, line_data.decimals, line_data.zero_tol)
        j = line_data.key_to_id.get(key)
        if j is None:
            return None
        ids.append(int(j))
    if len(set(ids)) != 12:
        return None
    return tuple(sorted(ids))


def span_key(F: np.ndarray, decimals: int = 8) -> bytes:
    """Rotation-independent hash of the 4-space via its orthogonal projector."""
    P = F.T @ F
    iu = np.triu_indices(P.shape[0])
    return quantize(P[iu][None, :], decimals)[0].tobytes()


# -----------------------------------------------------------------------------
# Shell rule O
# -----------------------------------------------------------------------------

@dataclass
class ShellRule:
    name: str
    shell_in_O: Callable[[np.ndarray, float, float], bool]
    pointwise_direction_ok: Optional[Callable[[np.ndarray, float], np.ndarray]] = None


def strict2o_pointwise(U: np.ndarray, tol: float) -> np.ndarray:
    """Safe pointwise test for directions in the 48-element binary octahedral group."""
    U = np.asarray(U, float)
    if U.ndim == 1:
        U = U[None, :]
    # Unit vectors: max dot = 1 iff equal.  Using abs would incorrectly quotient sign;
    # TWO_O already contains both signs.
    best = np.max(U @ TWO_O.T, axis=1)
    return best >= 1.0 - tol


def strict2o_shell(R: np.ndarray, radius: float, tol: float) -> bool:
    if len(R) == 0:
        return True
    return bool(np.all(strict2o_pointwise(R, tol)))


def load_shell_rule(rule_name: str, module_path: Optional[str]) -> ShellRule:
    if module_path:
        p = Path(module_path)
        spec = importlib.util.spec_from_file_location("user_quaternionic_O", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import O-module: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "shell_in_O"):
            raise ValueError("O-module must define shell_in_O(R, radius, tol).")
        point = getattr(mod, "pointwise_direction_ok", None)
        return ShellRule(f"module:{p}", mod.shell_in_O, point)

    if rule_name == "strict2o":
        return ShellRule("strict2o", strict2o_shell, strict2o_pointwise)
    raise ValueError(f"Unknown O-rule: {rule_name}")


def unique_directions(U: np.ndarray, decimals: int) -> np.ndarray:
    if len(U) == 0:
        return np.empty((0, 4), dtype=float)
    V = np.asarray(U, float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    q = quantize(V, decimals)
    _, idx = np.unique(q, axis=0, return_index=True)
    return V[np.sort(idx)]


def projection_shells(Q: np.ndarray,
                      zero_tol: float,
                      radius_tol: float,
                      direction_decimals: int) -> List[Tuple[float, np.ndarray, int]]:
    """Group nonzero 4D projections by radius; return (radius, distinct dirs, multiplicity)."""
    r = np.linalg.norm(Q, axis=1)
    nz = r > zero_tol
    if not np.any(nz):
        return []
    rn = r[nz]
    # Radius binning is only for identifying exact/repeated shells in numerical input.
    keys = np.rint(rn / radius_tol).astype(np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    ids0 = np.flatnonzero(nz)[order]
    shells = []
    a = 0
    while a < len(keys):
        b = a + 1
        while b < len(keys) and keys[b] == keys[a]:
            b += 1
        ids = ids0[a:b]
        rr = float(np.mean(r[ids]))
        U = Q[ids] / r[ids, None]
        R = unique_directions(U, direction_decimals)
        shells.append((rr, R, len(ids)))
        a = b
    return shells


def validate_projection(X: np.ndarray,
                        F: np.ndarray,
                        rule: ShellRule,
                        zero_tol: float,
                        shell_tol: float,
                        radius_tol: float,
                        direction_decimals: int,
                        sample_ids: Optional[np.ndarray] = None,
                        full: bool = True) -> Tuple[bool, dict]:
    """
    Validate the projection to span(F).

    If sample_ids is supplied and the rule has a pointwise safe prefilter, we
    can reject before full projection.  A passing sample is never accepted as
    proof: the full test is still performed when full=True.
    """
    if sample_ids is not None and rule.pointwise_direction_ok is not None:
        Qs = X[sample_ids] @ F.T
        rs = np.linalg.norm(Qs, axis=1)
        nz = rs > zero_tol
        if np.any(nz):
            U = Qs[nz] / rs[nz, None]
            ok = np.asarray(rule.pointwise_direction_ok(U, shell_tol))
            if ok.ndim == 0:
                ok = np.full(len(U), bool(ok))
            if not np.all(ok):
                return False, {
                    "stage": "sample_reject",
                    "sample_size": int(len(sample_ids)),
                    "bad_sample_directions": int(np.count_nonzero(~ok)),
                }
        if not full:
            return True, {"stage": "sample_pass", "sample_size": int(len(sample_ids))}

    Q = X @ F.T
    shells = projection_shells(Q, zero_tol, radius_tol, direction_decimals)
    shell_info = []
    for radius, R, multiplicity in shells:
        ok = bool(rule.shell_in_O(R, radius, shell_tol))
        shell_info.append({
            "radius": float(radius),
            "radius_squared": float(radius * radius),
            "projected_points": int(multiplicity),
            "distinct_directions": int(len(R)),
            "in_O": ok,
        })
        if not ok:
            return False, {
                "stage": "full_reject",
                "number_of_shells": len(shells),
                "shells": shell_info,
            }
    return True, {
        "stage": "full_pass",
        "zero_projections": int(np.count_nonzero(np.linalg.norm(Q, axis=1) <= zero_tol)),
        "number_of_shells": int(len(shells)),
        "shells": shell_info,
    }


# -----------------------------------------------------------------------------
# Candidate frame generation
# -----------------------------------------------------------------------------

def choose_basis_pool(line_data: LineData,
                      d: int,
                      mode: str,
                      cap: int) -> Tuple[np.ndarray, dict]:
    L = len(line_data.lines)
    if mode == "exhaustive":
        idx = np.arange(L, dtype=np.int32)
        return idx, {
            "heuristic": False,
            "support_threshold": None,
            "pool_size": int(L),
            "pool_is_all_lines": True,
        }

    # Auto: smallest support threshold containing at least d lines.
    vals = np.unique(line_data.supports)
    threshold = int(vals[-1])
    for s in vals:
        if np.count_nonzero(line_data.supports <= s) >= d:
            threshold = int(s)
            break
    idx = np.flatnonzero(line_data.supports <= threshold).astype(np.int32)
    uncapped = len(idx)
    if len(idx) > cap:
        # Deterministic: lowest support first, then line index.
        order = np.lexsort((idx, line_data.supports[idx]))
        idx = idx[order[:cap]]
    return idx, {
        "heuristic": True,
        "support_threshold": threshold,
        "pool_size_before_cap": int(uncapped),
        "pool_size": int(len(idx)),
        "pool_is_all_lines": bool(len(idx) == L),
        "cap": int(cap),
    }


def iter_k4_bitset(C: np.ndarray, orth_tol: float) -> Iterator[Tuple[int, int, int, int]]:
    """Enumerate 4-cliques in the orthogonality graph of rows of C."""
    m = len(C)
    if m < 4:
        return
    O = np.abs(C @ C.T) <= orth_tol
    np.fill_diagonal(O, False)
    adj = []
    for i in range(m):
        mask = 0
        for j in np.flatnonzero(O[i]):
            mask |= 1 << int(j)
        adj.append(mask)

    allmask = (1 << m) - 1
    for a in range(m):
        ma = adj[a] & (allmask ^ ((1 << (a + 1)) - 1))
        while ma:
            lb = ma & -ma
            b = lb.bit_length() - 1
            ma ^= lb
            mb = adj[a] & adj[b] & (allmask ^ ((1 << (b + 1)) - 1))
            while mb:
                lc = mb & -mb
                c = lc.bit_length() - 1
                mb ^= lc
                mc = adj[a] & adj[b] & adj[c] & (allmask ^ ((1 << (c + 1)) - 1))
                while mc:
                    ld = mc & -mc
                    e = ld.bit_length() - 1
                    mc ^= ld
                    yield (a, b, c, e)


def anchor_order(line_data: LineData, basis_idx: np.ndarray) -> np.ndarray:
    """Low-support anchors first; basis-pool lines get a slight priority."""
    L = len(line_data.lines)
    in_pool = np.zeros(L, dtype=np.int8)
    in_pool[basis_idx] = 1
    ids = np.arange(L, dtype=np.int32)
    # primary support, then pool first, then index
    return ids[np.lexsort((ids, -in_pool, line_data.supports))]


# -----------------------------------------------------------------------------
# Online orthogonality graph and clique search
# -----------------------------------------------------------------------------

@dataclass
class Cell:
    F: np.ndarray                     # 4 x d, orthonormal rows
    span_hash: bytes
    source: str
    anchor_line: int
    frame_lines: Tuple[int, int, int, int]
    projection_info: dict = field(default_factory=dict)


def spaces_orthogonal(F: np.ndarray, G: np.ndarray, tol: float) -> bool:
    return bool(np.max(np.abs(F @ G.T)) <= tol)


def find_clique_in_mask(adj: List[int], mask: int, need: int) -> Optional[List[int]]:
    """Find one clique of size need in an induced bit-mask graph."""
    if need == 0:
        return []
    if mask.bit_count() < need:
        return None
    if need == 1:
        v = (mask & -mask).bit_length() - 1
        return [v]

    # Recursive intersection.  k<=7 in the target applications.
    cand = mask
    while cand:
        lv = cand & -cand
        v = lv.bit_length() - 1
        cand ^= lv
        # Only vertices still after v in this branch, preventing repeats.
        sub = cand & adj[v]
        if sub.bit_count() >= need - 1:
            tail = find_clique_in_mask(adj, sub, need - 1)
            if tail is not None:
                return [v] + tail
    return None


class OnlineCellGraph:
    def __init__(self, k: int, orth_tol: float):
        self.k = k
        self.orth_tol = orth_tol
        self.cells: List[Cell] = []
        self.adj: List[int] = []

    def add(self, cell: Cell) -> Optional[List[int]]:
        n = len(self.cells)
        mask = 0
        for j, old in enumerate(self.cells):
            if spaces_orthogonal(cell.F, old.F, self.orth_tol):
                mask |= 1 << j
                self.adj[j] |= 1 << n
        self.cells.append(cell)
        self.adj.append(mask)

        if self.k == 1:
            return [n]
        if mask.bit_count() < self.k - 1:
            return None
        old_clique = find_clique_in_mask(self.adj, mask, self.k - 1)
        if old_clique is None:
            return None
        return old_clique + [n]


# -----------------------------------------------------------------------------
# Main search
# -----------------------------------------------------------------------------

def deterministic_sample_ids(X: np.ndarray, m: int) -> np.ndarray:
    n = len(X)
    if m <= 0 or m >= n:
        return np.arange(n, dtype=np.int32)
    # Mix evenly spaced rows with high-support rows.  No randomness needed.
    half = m // 2
    even = np.linspace(0, n - 1, max(1, half), dtype=np.int32)
    supp = np.sum(np.abs(X) > 1e-10, axis=1)
    high = np.argsort(supp, kind="stable")[-(m - len(even)):].astype(np.int32)
    return np.unique(np.concatenate([even, high])).astype(np.int32)


def make_cell(F: np.ndarray,
              source: str,
              anchor_line: int,
              frame_lines: Sequence[int],
              projection_info: dict,
              span_decimals: int) -> Cell:
    # Reorthonormalize defensively while preserving the span.  For an already
    # orthonormal frame this changes only at floating roundoff scale.
    Q, _ = np.linalg.qr(F.T)
    F2 = Q[:, :4].T
    return Cell(F2, span_key(F2, span_decimals), source, int(anchor_line),
                tuple(map(int, frame_lines)), projection_info)


def search(X: np.ndarray,
           rule: ShellRule,
           mode: str,
           line_decimals: int,
           span_decimals: int,
           zero_tol: float,
           orth_tol: float,
           half_tol: float,
           shell_tol: float,
           radius_tol: float,
           direction_decimals: int,
           basis_pool_cap: int,
           screen_rows: int,
           max_anchors: Optional[int],
           max_frames: Optional[int],
           progress_every: int,
           verbose: bool = True) -> Tuple[Optional[List[Cell]], dict]:
    n, d = X.shape
    k = d // 4
    t0 = time.time()

    if verbose:
        print(f"Loaded X: N={n}, d={d}, k={k}", flush=True)
        print("Extracting antipodal lines by hashing...", flush=True)

    ld = extract_antipodal_lines(X, line_decimals, zero_tol, require_both_signs=True)
    if verbose:
        print(f"Antipodal lines retained: {len(ld.lines)}", flush=True)
        u, c = np.unique(ld.supports, return_counts=True)
        print("Line support histogram:", dict(zip(map(int, u), map(int, c))), flush=True)

    basis_idx, pool_info = choose_basis_pool(ld, d, mode, basis_pool_cap)
    B = ld.lines[basis_idx]
    if verbose:
        print("Basis pool:", pool_info, flush=True)

    sample_ids = deterministic_sample_ids(X, screen_rows)
    order = anchor_order(ld, basis_idx)
    if max_anchors is not None:
        order = order[:max_anchors]

    graph = OnlineCellGraph(k, orth_tol)
    seen_cells = set()
    accepted_spans = set()
    tested_frames = 0
    certified_cells = 0
    projection_rejects = 0
    duplicate_spans = 0
    anchors_used = 0

    for ia, hidx0 in enumerate(order):
        hidx = int(hidx0)
        h = ld.lines[hidx]
        dots = B @ h
        cand_local = np.flatnonzero(np.abs(np.abs(dots) - 0.5) <= half_tol)
        if len(cand_local) < 4:
            continue
        C = B[cand_local]
        anchors_used += 1

        for q in iter_k4_bitset(C, orth_tol):
            tested_frames += 1
            if max_frames is not None and tested_frames > max_frames:
                complete = (mode == "exhaustive" and len(basis_idx) == len(ld.lines))
                return None, {
                    "status": "frame_limit",
                    "complete": False,
                    "elapsed_seconds": time.time() - t0,
                    "line_count": len(ld.lines),
                    "basis_pool": pool_info,
                    "anchors_used": anchors_used,
                    "frames_tested": tested_frames,
                    "certified_24cell_spans": certified_cells,
                    "admissible_vertices": len(graph.cells),
                    "projection_rejects": projection_rejects,
                    "duplicate_spans": duplicate_spans,
                }

            frame_local = np.asarray(q, dtype=int)
            frame_global = basis_idx[cand_local[frame_local]]
            F = ld.lines[frame_global]

            # The anchor is automatically a half-sign vector relative to F
            # (unit norm + four |dot|=1/2), but the remaining cell vertices
            # must still be present in X.
            cell_key = certify_24cell_frame(F, ld, orth_tol)
            if cell_key is None:
                continue
            if cell_key in seen_cells:
                duplicate_spans += 1
                continue
            seen_cells.add(cell_key)
            certified_cells += 1
            skey = span_key(F, span_decimals)

            # Safe early rejection when the O-rule supplies a pointwise test.
            ok, pinfo = validate_projection(
                X, F, rule, zero_tol, shell_tol, radius_tol,
                direction_decimals, sample_ids=sample_ids, full=False
            )
            if not ok:
                projection_rejects += 1
                continue

            ok, pinfo = validate_projection(
                X, F, rule, zero_tol, shell_tol, radius_tol,
                direction_decimals, sample_ids=None, full=True
            )
            if not ok:
                projection_rejects += 1
                continue

            # Once one admissible cell in a 4-space has been accepted, another
            # admissible gauge in the same 4-space cannot help the orthogonality
            # clique search, so it is safe to collapse them here (not before the
            # shell test).
            if skey in accepted_spans:
                continue
            accepted_spans.add(skey)

            cell = make_cell(F, "anchor-half-frame", hidx, frame_global,
                             pinfo, span_decimals)
            # make_cell QR can change basis orientation.  For shell rules tied
            # to the selected Hurwitz gauge this is undesirable; keep the
            # certified frame exactly (it is already orthonormal).
            cell.F = F.copy()
            cell.span_hash = skey
            clique_ids = graph.add(cell)

            if verbose and (len(graph.cells) <= 10 or len(graph.cells) % 100 == 0):
                print(
                    f"admissible vertex {len(graph.cells)}: anchor={hidx}, "
                    f"shells={pinfo.get('number_of_shells')}, "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

            if clique_ids is not None:
                cells = [graph.cells[j] for j in clique_ids]
                # Full final recheck of all k blocks.
                final_infos = []
                final_ok = True
                for c in cells:
                    vok, vinfo = validate_projection(
                        X, c.F, rule, zero_tol, shell_tol, radius_tol,
                        direction_decimals, sample_ids=None, full=True
                    )
                    final_infos.append(vinfo)
                    final_ok &= vok
                Trows = np.vstack([c.F for c in cells])
                orth_err = float(np.max(np.abs(Trows @ Trows.T - np.eye(d))))
                final_ok &= orth_err <= max(10 * orth_tol, 1e-5)
                if final_ok:
                    for c, info in zip(cells, final_infos):
                        c.projection_info = info
                    meta = {
                        "status": "success",
                        "complete": True,  # positive certificate is definitive
                        "elapsed_seconds": time.time() - t0,
                        "line_count": int(len(ld.lines)),
                        "basis_pool": pool_info,
                        "anchors_used": int(anchors_used),
                        "frames_tested": int(tested_frames),
                        "certified_24cell_spans": int(certified_cells),
                        "admissible_vertices": int(len(graph.cells)),
                        "projection_rejects": int(projection_rejects),
                        "duplicate_spans": int(duplicate_spans),
                        "clique_vertex_ids": list(map(int, clique_ids)),
                        "orthogonality_error": orth_err,
                    }
                    return cells, meta

        if verbose and progress_every > 0 and (ia + 1) % progress_every == 0:
            print(
                f"anchors {ia+1}/{len(order)} | frames={tested_frames} | "
                f"24cell-spans={certified_cells} | admissible={len(graph.cells)} | "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    enumeration_complete = bool(mode == "exhaustive" and len(basis_idx) == len(ld.lines))
    return None, {
        "status": "not_found",
        "complete": enumeration_complete,
        "elapsed_seconds": time.time() - t0,
        "line_count": int(len(ld.lines)),
        "basis_pool": pool_info,
        "anchors_used": int(anchors_used),
        "frames_tested": int(tested_frames),
        "certified_24cell_spans": int(certified_cells),
        "admissible_vertices": int(len(graph.cells)),
        "projection_rejects": int(projection_rejects),
        "duplicate_spans": int(duplicate_spans),
    }


# -----------------------------------------------------------------------------
# Certificate
# -----------------------------------------------------------------------------

def write_success(prefix: Path,
                  input_path: Path,
                  X: np.ndarray,
                  cells: List[Cell],
                  meta: dict,
                  rule: ShellRule,
                  save_rotated: bool) -> None:
    d = X.shape[1]
    Trows = np.vstack([c.F for c in cells])       # y = X @ Trows.T
    transform_path = prefix.with_name(prefix.name + "_transform.txt")
    cert_path = prefix.with_name(prefix.name + "_certificate.json")
    np.savetxt(transform_path, Trows.T, fmt="%.17g")

    block_data = []
    for j, c in enumerate(cells):
        block_data.append({
            "block": j,
            "anchor_line": c.anchor_line,
            "frame_line_ids": list(c.frame_lines),
            "basis_rows": c.F.tolist(),
            "projection": c.projection_info,
        })

    cert = {
        "result": "QUATERNIONIC_CONSTRUCTION_FOUND",
        "input": str(input_path),
        "N": int(X.shape[0]),
        "dimension": int(d),
        "k": int(d // 4),
        "O_rule": rule.name,
        "transform_convention": "X_rotated = X @ transform; transform columns are the selected frame vectors",
        "search": meta,
        "blocks": block_data,
    }
    cert_path.write_text(json.dumps(cert, indent=2), encoding="utf-8")

    if save_rotated:
        rot_path = prefix.with_name(prefix.name + "_rotated.txt")
        np.savetxt(rot_path, X @ Trows.T, fmt="%.17g")

    print("\nPASS: a quaternionic decomposition was found.")
    print(f"certificate: {cert_path}")
    print(f"transform:   {transform_path}")
    if save_rotated:
        print(f"rotated:     {rot_path}")


def write_failure(prefix: Path,
                  input_path: Path,
                  X: np.ndarray,
                  meta: dict,
                  rule: ShellRule) -> None:
    cert_path = prefix.with_name(prefix.name + "_search_report.json")
    report = {
        "result": "NOT_FOUND" if not meta.get("complete") else "NO_DECOMPOSITION_FOUND",
        "input": str(input_path),
        "N": int(X.shape[0]),
        "dimension": int(X.shape[1]),
        "k": int(X.shape[1] // 4),
        "O_rule": rule.name,
        "search": meta,
        "interpretation": (
            "Search exhausted all antipodal frame lines under this enumeration."
            if meta.get("complete") else
            "Fast/limited search did not find a certificate. This is NOT a proof of nonexistence."
        ),
    }
    cert_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if meta.get("complete"):
        print("\nFAIL: exhaustive enumeration found no quaternionic decomposition.")
    else:
        print("\nNOT FOUND: no certificate was found in the fast/limited search.")
        print("This is not a proof of nonexistence; use --mode exhaustive for completeness.")
    print(f"report: {cert_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Find/certify a quaternionic 24-cell decomposition of X in R^(4k)."
    )
    ap.add_argument("input", help="N x (4k) text coordinate matrix or .npy")
    ap.add_argument("--mode", choices=["auto", "exhaustive"], default="auto")
    ap.add_argument("--O-rule", choices=["strict2o"], default="strict2o",
                    help="built-in projection-shell rule")
    ap.add_argument("--O-module", default=None,
                    help="Python file defining shell_in_O(R,radius,tol), optionally pointwise_direction_ok")
    ap.add_argument("--prefix", default=None,
                    help="output prefix (default: INPUT_STEM_quat)")
    ap.add_argument("--save-rotated", action="store_true")

    ap.add_argument("--line-decimals", type=int, default=8,
                    help="decimal quantization for antipodal line hashing")
    ap.add_argument("--span-decimals", type=int, default=8)
    ap.add_argument("--direction-decimals", type=int, default=8)
    ap.add_argument("--zero-tol", type=float, default=2e-7)
    ap.add_argument("--orth-tol", type=float, default=3e-6)
    ap.add_argument("--half-tol", type=float, default=3e-6)
    ap.add_argument("--shell-tol", type=float, default=4e-6)
    ap.add_argument("--radius-tol", type=float, default=2e-6)

    ap.add_argument("--basis-pool-cap", type=int, default=4000,
                    help="auto-mode cap for low-support possible frame lines")
    ap.add_argument("--screen-rows", type=int, default=1024,
                    help="safe early pointwise screening sample when O-rule supports it")
    ap.add_argument("--max-anchors", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=250)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    prefix = Path(args.prefix) if args.prefix else inp.with_name(inp.stem + "_quat")
    X = load_coordinates(inp)
    rule = load_shell_rule(args.O_rule, args.O_module)

    cells, meta = search(
        X=X,
        rule=rule,
        mode=args.mode,
        line_decimals=args.line_decimals,
        span_decimals=args.span_decimals,
        zero_tol=args.zero_tol,
        orth_tol=args.orth_tol,
        half_tol=args.half_tol,
        shell_tol=args.shell_tol,
        radius_tol=args.radius_tol,
        direction_decimals=args.direction_decimals,
        basis_pool_cap=args.basis_pool_cap,
        screen_rows=args.screen_rows,
        max_anchors=args.max_anchors,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
        verbose=True,
    )

    if cells is not None:
        write_success(prefix, inp, X, cells, meta, rule, args.save_rotated)
    else:
        write_failure(prefix, inp, X, meta, rule)


if __name__ == "__main__":
    main()
