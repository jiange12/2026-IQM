from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import deque
# ----------------------------
# Parsing IQM metrics JSON
# ----------------------------

EDGE_PAIR_RE = re.compile(r"(QB\d+)__(QB\d+)")


@dataclass(frozen=True)
class NodeMetrics:
    t1_s: Optional[float] = None
    t2_s: Optional[float] = None
    readout_e01: Optional[float] = None
    readout_e10: Optional[float] = None
    readout_fid: Optional[float] = None


@dataclass(frozen=True)
class EdgeMetrics:
    uz_cz_fid: Optional[float] = None
    irb_crf_crf_fid: Optional[float] = None



def reachable_node_count(adjacency: Dict[str, List[Tuple[str, float]]], start: str) -> int:
    """How many nodes are reachable at all (ignoring costs)."""
    seen = set([start])
    q = deque([start])
    while q:
        a = q.popleft()
        for b, _ in adjacency.get(a, []):
            if b not in seen:
                seen.add(b)
                q.append(b)
    return len(seen)


def find_best_start_for_fixed_length(
    adjacency: Dict[str, List[Tuple[str, float]]],
    node_cost: Dict[str, float],
    *,
    n_nodes: int = 10,
    exp_gamma: float = 0.85,
    iterations_per_start: int = 6000,
    exploration_c: float = 1.4,
    rollout_bias: float = 0.8,
    simple_path: bool = True,
    seed: int = 0,
    verbose_every: int = 10,
    raise_on_fail: bool = True,
    beam_width: int = 2000,
    beam_restarts: int = 5,
) -> Tuple[str, List[str], float]:
    """
    Try MCTS starting from *every* node in adjacency, return the best (lowest-cost)
    length-n_nodes path found.

    Returns:
      best_start, best_path, best_cost
    """
    all_nodes = sorted(adjacency.keys())
    best_start = None
    best_path: List[str] = []
    best_cost = float("inf")

    skipped = 0
    failed = 0

    for idx, start in enumerate(all_nodes):
        # Quick feasibility: if we require a simple path of length n_nodes,
        # we must have at least n_nodes reachable distinct nodes.
        if simple_path:
            if reachable_node_count(adjacency, start) < n_nodes:
                skipped += 1
                continue

        try:
            # Different seed per start for diversity but reproducibility
            local_seed = (seed * 1000003 + hash(start)) & 0xFFFFFFFF

            path, cost = mcts_find_min_cost_path(
                adjacency,
                node_cost,
                start=start,
                goal=None,
                n_nodes=n_nodes,
                iterations=iterations_per_start,
                exploration_c=exploration_c,
                rollout_bias=rollout_bias,
                simple_path=simple_path,
                seed=local_seed,
                exp_gamma=exp_gamma,
                beam_width=beam_width,
                beam_restarts=beam_restarts,
            )

            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_start = start

        except Exception:
            failed += 1

        if verbose_every and (idx + 1) % verbose_every == 0:
            print(
                f"[sweep {idx+1}/{len(all_nodes)}] current best: "
                f"{best_start} cost={best_cost:.6f} | skipped={skipped} failed={failed}"
            )

    # If we didn't find any valid path, try a targeted retry instead of hard-failing.
    # This helps when a path exists but the search budget was too small or unlucky.
    if best_start is None or not best_path:
        # Retry only on the starts with the largest reachable set.
        if simple_path and all_nodes:
            reach = [(reachable_node_count(adjacency, s), s) for s in all_nodes]
            reach.sort(reverse=True)
            retry_starts = [s for _, s in reach[: min(12, len(reach))]]
        else:
            retry_starts = all_nodes

        retry_best_start = None
        retry_best_path: List[str] = []
        retry_best_cost = float("inf")

        # Increase the per-start budget for the retry, but keep it bounded.
        retry_iters = max(iterations_per_start * 5, iterations_per_start + 5000)

        for start in retry_starts:
            if simple_path and reachable_node_count(adjacency, start) < n_nodes:
                continue
            try:
                local_seed = (seed * 1000003 + hash((start, "retry"))) & 0xFFFFFFFF
                path, cost = mcts_find_min_cost_path(
                    adjacency,
                    node_cost,
                    start=start,
                    goal=None,
                    n_nodes=n_nodes,
                    iterations=retry_iters,
                    exploration_c=exploration_c,
                    rollout_bias=rollout_bias,
                    simple_path=simple_path,
                    seed=local_seed,
                    exp_gamma=exp_gamma,
                )
                if cost < retry_best_cost:
                    retry_best_cost = cost
                    retry_best_path = path
                    retry_best_start = start
            except Exception:
                continue

        if retry_best_start is not None and retry_best_path:
            return retry_best_start, retry_best_path, retry_best_cost

        if raise_on_fail:
            raise RuntimeError(
                f"No valid length-{n_nodes} path found from any start. "
                f"(skipped={skipped}, failed={failed})"
            )
        return "", [], float("inf")

    return best_start, best_path, best_cost


def find_best_gamma_for_fixed_length(
    adjacency: Dict[str, List[Tuple[str, float]]],
    node_cost: Dict[str, float],
    *,
    start: Optional[str],
    goal: Optional[str],
    n_nodes: int,
    gamma_min: float = 0.55,
    gamma_max: float = 0.99,
    gamma_steps: int = 12,
    sweep_starts: bool = False,
    iterations: int = 6000,
    exploration_c: float = 1.4,
    rollout_bias: float = 0.8,
    simple_path: bool = True,
    seed: int = 0,
    verbose: bool = True,
    beam_width: int = 2000,
    beam_restarts: int = 5,
) -> Tuple[float, str, List[str], float]:
    """Grid-search the exponential position-weight base exp_gamma.

    We search exp_gamma in (0,1] on a *log-spaced* grid (i.e., exponential grid)
    so that we more evenly cover the range of decay rates.

    Returns:
      best_gamma, best_start, best_path, best_cost
    """
    if gamma_steps < 2:
        raise ValueError("gamma_steps must be >= 2")
    if not (0.0 < gamma_min < gamma_max <= 1.0):
        raise ValueError("Require 0 < gamma_min < gamma_max <= 1")

    # Log-spaced grid in gamma
    log_min = math.log(gamma_min)
    log_max = math.log(gamma_max)
    gamma_vals = [math.exp(log_min + i * (log_max - log_min) / (gamma_steps - 1)) for i in range(gamma_steps)]

    best_gamma = float("nan")
    best_start = ""
    best_path: List[str] = []
    best_cost = float("inf")

    for idx, gam in enumerate(gamma_vals):
        if sweep_starts:
            cand_start, cand_path, cand_cost = find_best_start_for_fixed_length(
                adjacency,
                node_cost,
                n_nodes=n_nodes,
                exp_gamma=gam,
                iterations_per_start=iterations,
                exploration_c=exploration_c,
                rollout_bias=rollout_bias,
                simple_path=simple_path,
                seed=seed,
                verbose_every=0,
                raise_on_fail=False,
            )
        else:
            if start is None:
                raise ValueError("start must be provided when sweep_starts=False")
            cand_path, cand_cost = mcts_find_min_cost_path(
                adjacency,
                node_cost,
                start=start,
                goal=goal,
                n_nodes=n_nodes,
                iterations=iterations,
                exploration_c=exploration_c,
                rollout_bias=rollout_bias,
                simple_path=simple_path,
                seed=seed,
                exp_gamma=gam,
            )
            cand_start = start

        if cand_cost < best_cost:
            best_cost = cand_cost
            best_gamma = float(gam)
            best_start = cand_start
            best_path = cand_path

        if verbose:
            print(f"[gamma {idx+1}/{len(gamma_vals)}] gamma={gam:.6f} best_cost={best_cost:.6f}")

    if not best_path:
        # No gamma value yielded a feasible path. Return a sentinel result so callers
        # (e.g., --sweep_n) can skip this n instead of crashing.
        return float("nan"), "", [], float("inf")

    return best_gamma, best_start, best_path, best_cost

def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _find_observations(obj: Any) -> List[Dict[str, Any]]:
    """
    IQM exports vary. We search common containers for a list of dict items.
    Each observation is expected to include at least:
      - dut_field: str
      - value: number-like
      - invalid: bool (optional)
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key_str in ("observations", "data", "items", "results"):
            if key_str in obj and isinstance(obj[key_str], list):
                return [x for x in obj[key_str] if isinstance(x, dict)]

    raise ValueError("Could not locate an observations list in JSON. Expected keys like observations/data/items/results.")


def parse_iqm_metrics_json(path_str: str) -> Tuple[Dict[str, NodeMetrics], Dict[Tuple[str, str], EdgeMetrics]]:
    """
    Extracts:
      - node metrics: T1, T2, readout errors/fidelity
      - edge metrics: 2Q fidelities for connected qubit pairs (QB*__QB*)

    Returns:
      node_metrics: QBxx -> NodeMetrics
      edge_metrics: (QBxx, QByy) with QBxx < QByy -> EdgeMetrics
    """
    with open(path_str, "r") as fp:
        raw = json.load(fp)

    observations = _find_observations(raw)

    node_metrics: Dict[str, NodeMetrics] = {}
    edge_metrics: Dict[Tuple[str, str], EdgeMetrics] = {}

    def get_node(qb_name: str) -> NodeMetrics:
        return node_metrics.get(qb_name, NodeMetrics())

    def set_node(qb_name: str, nm: NodeMetrics) -> None:
        node_metrics[qb_name] = nm

    def norm_edge(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def get_edge(a: str, b: str) -> EdgeMetrics:
        return edge_metrics.get(norm_edge(a, b), EdgeMetrics())

    def set_edge(a: str, b: str, em: EdgeMetrics) -> None:
        edge_metrics[norm_edge(a, b)] = em

    for obs in observations:
        if obs.get("invalid", False):
            continue

        dut_field = obs.get("dut_field", "")
        val = _safe_float(obs.get("value"))

        if not dut_field or val is None:
            continue

        # ---- Node metrics ----
        # characterization.model.QB6.t1_time
        m = re.search(r"characterization\.model\.(QB\d+)\.t1_time$", dut_field)
        if m:
            qb = m.group(1)
            nm = get_node(qb)
            set_node(qb, NodeMetrics(
                t1_s=val,
                t2_s=nm.t2_s,
                readout_e01=nm.readout_e01,
                readout_e10=nm.readout_e10,
                readout_fid=nm.readout_fid,
            ))
            continue

        # characterization.model.QB5.t2_echo_time OR characterization.model.QB35.t2_time
        m = re.search(r"characterization\.model\.(QB\d+)\.(t2_echo_time|t2_time)$", dut_field)
        if m:
            qb = m.group(1)
            nm = get_node(qb)
            set_node(qb, NodeMetrics(
                t1_s=nm.t1_s,
                t2_s=val,
                readout_e01=nm.readout_e01,
                readout_e10=nm.readout_e10,
                readout_fid=nm.readout_fid,
            ))
            continue

        # metrics.ssro.measure_fidelity.constant.QB21.error_0_to_1 / error_1_to_0
        m = re.search(r"metrics\.ssro\.measure_fidelity\.constant\.(QB\d+)\.(error_0_to_1|error_1_to_0)$", dut_field)
        if m:
            qb, which = m.group(1), m.group(2)
            nm = get_node(qb)
            e01 = val if which == "error_0_to_1" else nm.readout_e01
            e10 = val if which == "error_1_to_0" else nm.readout_e10
            set_node(qb, NodeMetrics(
                t1_s=nm.t1_s,
                t2_s=nm.t2_s,
                readout_e01=e01,
                readout_e10=e10,
                readout_fid=nm.readout_fid,
            ))
            continue

        # metrics.ssro.measure.constant.QB20.fidelity OR metrics.ssro.measure_fidelity.constant.QB18.fidelity
        m = re.search(r"metrics\.ssro\.(measure|measure_fidelity)\.constant\.(QB\d+)\.fidelity$", dut_field)
        if m:
            qb = m.group(2)
            nm = get_node(qb)
            set_node(qb, NodeMetrics(
                t1_s=nm.t1_s,
                t2_s=nm.t2_s,
                readout_e01=nm.readout_e01,
                readout_e10=nm.readout_e10,
                readout_fid=val,
            ))
            continue

        # ---- Edge metrics ----
        # metrics.rb.clifford.uz_cz.QB4__QB5.fidelity:par=d2
        m = re.search(r"metrics\.rb\.clifford\.uz_cz\.(QB\d+__QB\d+)\.fidelity(?::par=.*)?$", dut_field)
        if m:
            a, b = m.group(1).split("__")
            em = get_edge(a, b)
            set_edge(a, b, EdgeMetrics(
                uz_cz_fid=val,
                irb_crf_crf_fid=em.irb_crf_crf_fid,
            ))
            continue

        # metrics.irb.cz.crf_crf.QB4__QB5.fidelity:par=d2
        m = re.search(r"metrics\.irb\.cz\.crf_crf\.(QB\d+__QB\d+)\.fidelity(?::par=.*)?$", dut_field)
        if m:
            a, b = m.group(1).split("__")
            em = get_edge(a, b)
            set_edge(a, b, EdgeMetrics(
                uz_cz_fid=em.uz_cz_fid,
                irb_crf_crf_fid=val,
            ))
            continue

    return node_metrics, edge_metrics


# ----------------------------
# Build weights + adjacency
# ----------------------------

def node_metrics_to_metrics_by_node(node_metrics: Dict[str, NodeMetrics]) -> Dict[str, Dict[str, float]]:
    """
    Produces:
      metrics_by_node = {
        "QB46": {"T1_us": ..., "T2_us": ..., "readout_err": ...},
        ...
      }
    Missing fields become NaN.
    """
    out: Dict[str, Dict[str, float]] = {}

    for qb, nm in node_metrics.items():
        t1_us = nm.t1_s * 1e6 if nm.t1_s is not None else float("nan")
        t2_us = nm.t2_s * 1e6 if nm.t2_s is not None else float("nan")

        # Prefer explicit readout errors if present; else use (1 - fidelity)
        if nm.readout_e01 is not None or nm.readout_e10 is not None:
            e01 = nm.readout_e01 if nm.readout_e01 is not None else nm.readout_e10
            e10 = nm.readout_e10 if nm.readout_e10 is not None else nm.readout_e01
            if e01 is None or e10 is None:
                ro_err = float("nan")
            else:
                ro_err = 0.5 * (float(e01) + float(e10))
        elif nm.readout_fid is not None:
            ro_err = 1.0 - float(nm.readout_fid)
        else:
            ro_err = float("nan")

        out[qb] = {"T1_us": float(t1_us), "T2_us": float(t2_us), "readout_err": float(ro_err)}

    return out


def build_node_weights(
    metrics_by_node: Dict[str, Dict[str, float]],
    *,
    w_t1: float = 1.0,
    w_t2: float = 1.0,
    w_ro: float = 2.0,
    eps: float = 1e-9,
    missing_penalty: float = 1.0,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Produces scalar node *cost* (lower is better).

    Heuristic:
      cost += w_t1 * (1/T1_us)          (if known else penalty)
      cost += w_t2 * (1/T2_us)          (if known else penalty)
      cost += w_ro * (readout_err)      (if known else modest default)

    missing_penalty: added when T1 or T2 missing to bias away from unknown qubits.
    """
    raw: Dict[str, float] = {}

    for qb, met in metrics_by_node.items():
        t1 = met.get("T1_us", float("nan"))
        t2 = met.get("T2_us", float("nan"))
        ro = met.get("readout_err", float("nan"))

        cost = 0.0
        if math.isfinite(t1) and t1 > 0:
            cost += w_t1 * (1.0 / (t1 + eps))
        else:
            cost += w_t1 * missing_penalty

        if math.isfinite(t2) and t2 > 0:
            cost += w_t2 * (1.0 / (t2 + eps))
        else:
            cost += w_t2 * missing_penalty

        if math.isfinite(ro) and ro >= 0:
            cost += w_ro * ro
        else:
            cost += w_ro * 0.05  # mild fallback

        raw[qb] = float(cost)

    if not normalize:
        return raw

    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in raw}

    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def build_adjacency(
    edge_metrics: Dict[Tuple[str, str], EdgeMetrics],
    *,
    prefer: str = "worst",  # "uz_cz", "irb", or "worst"
    default_fidelity: float = 0.0,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Builds adjacency list with *edge cost* = 1 - fidelity.
    Lower cost is better.

    prefer:
      - "uz_cz": prefer uz_cz fidelity else irb
      - "irb": prefer irb fidelity else uz_cz
      - "worst": take min(uz_cz, irb) if both exist (most conservative)
    """
    adj: Dict[str, List[Tuple[str, float]]] = {}

    def pick_fidelity(em: EdgeMetrics) -> float:
        uz = em.uz_cz_fid
        irb = em.irb_crf_crf_fid
        if prefer == "uz_cz":
            if uz is not None:
                return float(uz)
            if irb is not None:
                return float(irb)
        elif prefer == "irb":
            if irb is not None:
                return float(irb)
            if uz is not None:
                return float(uz)
        else:  # "worst"
            candidates = [x for x in (uz, irb) if x is not None]
            if candidates:
                return float(min(candidates))

        return float(default_fidelity)

    for (a, b), em in edge_metrics.items():
        fid = pick_fidelity(em)
        cost = 1.0 - fid

        adj.setdefault(a, []).append((b, cost))
        adj.setdefault(b, []).append((a, cost))

    # Optional: sort neighbors by cost to help rollouts pick good edges
    for a in adj:
        adj[a].sort(key=lambda t: t[1])

    return adj


def prune_graph(
    node_weights: Dict[str, float],
    adjacency: Dict[str, List[Tuple[str, float]]],
    *,
    max_keep: Optional[int] = None,
    drop_if_weight_ge: Optional[float] = None,
    mandatory_keep: Optional[Iterable[str]] = None,
    required_min_nodes: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    """
    Heuristic pruning (safe for long simple paths):

      1) Start from the best nodes by node weight (lower is better), optionally limited by max_keep.
      2) Optionally drop nodes with weight >= drop_if_weight_ge.
      3) Always keep mandatory_keep nodes.
      4) **Safety constraint:** ensure at least required_min_nodes nodes survive pruning by
         minimally relaxing drop/max_keep as needed (adding back best-weight nodes).

    Notes:
      - This does NOT guarantee that every start has a reachable simple path of length n;
        connectivity is handled later by reachability checks per start.
      - Adjacency keys include isolated kept nodes (with empty neighbor lists) so node counts
        reflect what was kept.
    """
    mandatory = set(mandatory_keep or [])
    items = sorted(node_weights.items(), key=lambda kv: kv[1])  # (node, weight) ascending

    # Determine initial candidate pool size.
    candidate_count = len(items) if max_keep is None else int(max_keep)
    if required_min_nodes is not None:
        candidate_count = max(candidate_count, int(required_min_nodes))
    candidate_count = min(candidate_count, len(items))

    # Candidate pool from best-weight nodes
    candidate_nodes = [k for k, _ in items[:candidate_count]]
    keep: set[str] = set(candidate_nodes)

    # Apply hard threshold, if any
    if drop_if_weight_ge is not None:
        keep = {k for k in keep if node_weights[k] < drop_if_weight_ge}

    # Always keep mandatory nodes
    keep |= mandatory

    # Safety: if we pruned too aggressively for the requested path length, add back
    # best-weight nodes (globally) until we have enough.
    if required_min_nodes is not None:
        req = int(required_min_nodes)
        if len(keep) < req:
            for k, _ in items:
                if k not in keep:
                    keep.add(k)
                    if len(keep) >= req:
                        break

    # Build new structures. IMPORTANT: include isolated nodes as keys.
    new_weights = {k: node_weights[k] for k in keep if k in node_weights}
    new_adj: Dict[str, List[Tuple[str, float]]] = {k: [] for k in keep}

    for a in keep:
        for b, c in adjacency.get(a, []):
            if b in keep:
                new_adj[a].append((b, c))

    for a in new_adj:
        new_adj[a].sort(key=lambda t: t[1])

    return new_weights, new_adj


# ----------------------------
# MCTS for fixed-length path
# ----------------------------

@dataclass
class MCTSNode:
    # State
    current: str
    path: List[str]
    cost_so_far: float
    steps_remaining: int

    # Tree stats
    visits: int = 0
    total_reward: float = 0.0  # reward = -total_cost
    children: Dict[str, "MCTSNode"] = None
    untried_actions: List[str] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}
        if self.untried_actions is None:
            self.untried_actions = []


def exp_position_weight(pos_idx: int, exp_gamma: float) -> float:
    """Exponential position weight (earlier positions get higher weight).

    We use: w(pos) = exp_gamma ** pos, with exp_gamma in (0, 1].
      - pos=0 (start) => w=1
      - pos=1 => w=exp_gamma
      - ... decreasing geometrically
    """
    if pos_idx < 0:
        raise ValueError("pos_idx must be >= 0")
    if not (0.0 < exp_gamma <= 1.0):
        raise ValueError("exp_gamma must be in (0, 1]")
    return float(exp_gamma ** pos_idx)


def mcts_find_min_cost_path(
    adjacency: Dict[str, List[Tuple[str, float]]],
    node_cost: Dict[str, float],
    *,
    start: str,
    n_nodes: int,
    goal: Optional[str] = None,
    iterations: int = 20000,
    exploration_c: float = 1.4,
    rollout_bias: float = 0.8,
    simple_path: bool = True,
    seed: int = 0,
    exp_gamma: float = 0.85,
    beam_width: int = 2000,
    beam_restarts: int = 5,
) -> Tuple[List[str], float]:
    """
    Branch-and-Bound DFS solver (replaces the previous MCTS implementation).

    Find a path with exactly n_nodes nodes (steps = n_nodes-1 moves) while minimizing:

        sum(edge_costs along the path)
        + sum( w(pos) * node_cost(node_at_pos) for visited nodes )

    where position weights are exponential:
        w(pos) = exp_gamma ** pos

    If goal is not None, the path must end at goal at exactly length n_nodes.
    If goal is None, we minimize cost among any length-n_nodes path.

    Args retained for CLI compatibility:
      - iterations: interpreted as a cap on DFS node-expansions explored (approx. work limit).
      - exploration_c / rollout_bias: unused (legacy MCTS parameters kept for compatibility).
      - seed: used only for tie-breaking shuffle to diversify equivalent-cost branches.

    Returns:
      (best_path, best_cost)

    Notes on pruning bound:
      We compute an *optimistic* (lower) bound on the remaining cost:
        - first remaining step uses the cheapest available outgoing move from the current node
          (respecting simple_path for the immediate step).
        - subsequent remaining steps use globally minimal per-position incremental costs,
          ignoring reachability / path-coupling (still a valid lower bound, but optimistic).
      This bound is fast and typically strong enough on sparse lattices with varied weights.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    if start not in adjacency:
        raise ValueError(f"Start node {start} has no adjacency entries.")

    rng = random.Random(seed)

    # ---------- Build index mappings for speed ----------
    names = sorted(adjacency.keys())
    idx_of: Dict[str, int] = {nm: i for i, nm in enumerate(names)}
    if start not in idx_of:
        raise ValueError(f"Start node {start} missing from adjacency keys.")
    if goal is not None and goal not in idx_of:
        # goal may have been pruned; we treat as impossible
        return ([], float("inf"))

    V = len(names)
    node_cost_arr = [float(node_cost.get(nm, 0.0)) for nm in names]

    # adjacency list in index form
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(V)]
    all_edges: List[Tuple[int, int, float]] = []
    for a_nm, nbrs in adjacency.items():
        a = idx_of[a_nm]
        for b_nm, c in nbrs:
            if b_nm not in idx_of:
                continue
            b = idx_of[b_nm]
            cc = float(c)
            adj[a].append((b, cc))
            all_edges.append((a, b, cc))

    # quick feasibility: for a simple path we need at least n_nodes distinct reachable nodes
    if simple_path and reachable_node_count(adjacency, start) < n_nodes:
        return ([], float("inf"))

    steps_total = n_nodes - 1
    start_idx = idx_of[start]
    goal_idx = idx_of[goal] if goal is not None else -1

    # ---------- Precompute per-position weights ----------
    # pos index is the index in the path of the node we are *entering*.
    # start node is pos 0, first move enters pos 1, ..., last node is pos (n_nodes-1).
    w_by_pos = [exp_position_weight(pos, exp_gamma) for pos in range(n_nodes)]

    # ---------- Precompute optimistic global minima per position ----------
    # min incremental cost to "take a step into position pos" over all directed edges.
    # inc = edge_cost(a->b) + w(pos) * node_cost(b)
    if all_edges:
        global_min_inc = [0.0] * n_nodes
        global_min_inc[0] = 0.0  # no move to enter start
        for pos in range(1, n_nodes):
            wpos = w_by_pos[pos]
            best = float("inf")
            for _, b, ec in all_edges:
                cand = ec + (wpos * node_cost_arr[b])
                if cand < best:
                    best = cand
            global_min_inc[pos] = best if best != float("inf") else 0.0
    else:
        global_min_inc = [0.0] * n_nodes

    # suffix sum of global minima from pos to end (pos in [0..n_nodes])
    suffix_global = [0.0] * (n_nodes + 1)
    for pos in range(n_nodes - 1, 0, -1):
        suffix_global[pos] = suffix_global[pos + 1] + global_min_inc[pos]


    # ---------- Fast incumbent via beam search (seed for pruning) ----------
    inc_best_cost = float("inf")
    inc_best_path_idx: List[int] = []

    # Root cost includes the start node cost at position 0
    root_cost_local = w_by_pos[0] * node_cost_arr[start_idx]

    def _beam_search_once(restart_idx: int) -> Tuple[List[int], float]:
        """
        Beam search to quickly find a good (not necessarily optimal) simple path of length n_nodes.
        This produces a strong *upper bound* (incumbent) to accelerate branch-and-bound pruning.
        """
        if n_nodes == 1:
            return [start_idx], root_cost_local

        # Each state: (score_for_ranking, cost_so_far, last_node, path_list, visited_set)
        # score_for_ranking is optimistic: cost_so_far + optimistic remaining suffix
        beam: List[Tuple[float, float, int, List[int], set]] = []
        beam.append((root_cost_local + suffix_global[1], root_cost_local, start_idx, [start_idx], {start_idx}))

        # Small randomized tie-breaking
        local_rng = random.Random(seed + 10007 * (restart_idx + 1))

        for next_pos in range(1, n_nodes):
            new_states: List[Tuple[float, float, int, List[int], set]] = []
            wpos = w_by_pos[next_pos]
            for _, cost_so_far, cur, path_list, vis in beam:
                # neighbors
                nbrs = adj[cur]
                if not nbrs:
                    continue

                # enforce goal on final position if needed
                if goal is not None and next_pos == n_nodes - 1:
                    # only transitions into goal_idx are allowed
                    for nb, ec in nbrs:
                        if nb != goal_idx:
                            continue
                        if simple_path and nb in vis:
                            continue
                        inc = ec + (wpos * node_cost_arr[nb])
                        new_cost = cost_so_far + inc
                        new_path = path_list + [nb]
                        new_vis = set(vis)
                        new_vis.add(nb)
                        # optimistic remaining after this position
                        score = new_cost
                        new_states.append((score, new_cost, nb, new_path, new_vis))
                    continue

                # normal expansion
                # optional random shuffle for diversity among equal-weight edges
                nbrs2 = list(nbrs)
                if restart_idx > 0:
                    local_rng.shuffle(nbrs2)

                for nb, ec in nbrs2:
                    if simple_path and nb in vis:
                        continue
                    inc = ec + (wpos * node_cost_arr[nb])
                    new_cost = cost_so_far + inc
                    new_path = path_list + [nb]
                    new_vis = set(vis)
                    new_vis.add(nb)
                    # optimistic: add the global suffix for remaining positions (ignoring coupling)
                    score = new_cost + (suffix_global[next_pos + 1] if next_pos + 1 < len(suffix_global) else 0.0)
                    new_states.append((score, new_cost, nb, new_path, new_vis))

            if not new_states:
                return ([], float("inf"))

            # Keep top beam_width by optimistic score
            new_states.sort(key=lambda t: t[0])
            if beam_width and len(new_states) > beam_width:
                new_states = new_states[:beam_width]

            beam = new_states

        # pick best feasible complete path (min actual cost)
        best_state = min(beam, key=lambda t: t[1], default=None)
        if best_state is None:
            return ([], float("inf"))
        return best_state[3], best_state[1]

    if beam_width > 0 and beam_restarts > 0:
        # Keep restarts modest; beam is only to find an incumbent quickly.
        for r in range(max(1, beam_restarts)):
            cand_path_idx, cand_cost = _beam_search_once(r)
            if cand_path_idx and cand_cost < inc_best_cost:
                inc_best_cost = cand_cost
                inc_best_path_idx = cand_path_idx

    # ---------- DFS branch-and-bound ----------
    best_cost = inc_best_cost
    best_path_idx: List[int] = list(inc_best_path_idx)

    visited = [False] * V
    path_idx: List[int] = [start_idx]
    visited[start_idx] = True

    # Root cost includes the start node cost at position 0
    root_cost = w_by_pos[0] * node_cost_arr[start_idx]

    expansions = 0
    pruned = 0

    def lower_bound_additional(cur: int, next_pos: int, steps_remaining: int) -> float:
        """
        Optimistic lower bound on additional cost to complete 'steps_remaining' moves,
        where the next entered position will be 'next_pos'.
        """
        if steps_remaining <= 0:
            return 0.0

        # If we are on the last move and have a goal, enforce it tightly.
        if goal is not None and steps_remaining == 1:
            if visited[goal_idx]:
                return float("inf")
            wpos = w_by_pos[next_pos]
            best = float("inf")
            for nb, ec in adj[cur]:
                if nb == goal_idx and (not visited[nb]):
                    cand = ec + (wpos * node_cost_arr[nb])
                    if cand < best:
                        best = cand
            return best

        # First step: choose cheapest available outgoing step from cur (respecting visited for immediate move).
        wpos = w_by_pos[next_pos]
        best_first = float("inf")
        for nb, ec in adj[cur]:
            if simple_path and visited[nb]:
                continue
            cand = ec + (wpos * node_cost_arr[nb])
            if cand < best_first:
                best_first = cand

        if best_first == float("inf"):
            return float("inf")

        if steps_remaining == 1:
            return best_first

        # Remaining steps after the first are bounded by global optimistic minima.
        # next_pos+1 is the next entered position after taking the first step.
        return best_first + suffix_global[next_pos + 1]

    def dfs(cur: int, cost_so_far: float, steps_remaining: int) -> None:
        nonlocal best_cost, best_path_idx, expansions, pruned

        # Work cap
        if expansions >= iterations:
            return

        # Terminal
        if steps_remaining == 0:
            if goal is not None and cur != goal_idx:
                return
            if cost_so_far < best_cost:
                best_cost = cost_so_far
                best_path_idx = list(path_idx)
            return

        next_pos = len(path_idx)  # position of the next node to be appended
        # Bound check
        lb_add = lower_bound_additional(cur, next_pos, steps_remaining)
        if lb_add == float("inf"):
            return
        if cost_so_far + lb_add >= best_cost:
            pruned += 1
            return

        # Generate actions
        actions: List[Tuple[float, int]] = []
        wpos = w_by_pos[next_pos]

        for nb, ec in adj[cur]:
            if simple_path and visited[nb]:
                continue
            # If goal specified and we're on the last move, we must go to goal.
            if goal is not None and steps_remaining == 1 and nb != goal_idx:
                continue
            inc = ec + (wpos * node_cost_arr[nb])
            actions.append((inc, nb))

        if not actions:
            return

        # Explore cheaper branches first to tighten the upper bound quickly
        actions.sort(key=lambda t: t[0])

        # If there are many ties, shuffle a tiny bit for diversity (still mostly cost-ordered)
        # This helps when many equal-weight edges exist.
        if len(actions) > 1 and abs(actions[0][0] - actions[-1][0]) < 1e-15:
            rng.shuffle(actions)

        for inc, nb in actions:
            if expansions >= iterations:
                return

            # Additional prune using a cheap optimistic remainder after taking this action
            new_cost = cost_so_far + inc
            # Early cut if already worse
            if new_cost >= best_cost:
                pruned += 1
                continue

            # Apply move
            expansions += 1
            visited[nb] = True
            path_idx.append(nb)

            dfs(nb, new_cost, steps_remaining - 1)

            # Undo move
            path_idx.pop()
            visited[nb] = False

    dfs(start_idx, root_cost, steps_total)

    if not best_path_idx or best_cost == float("inf"):
        return ([], float("inf"))

    best_path = [names[i] for i in best_path_idx]
    return best_path, best_cost


def extract_greedy_best_path(
    root: MCTSNode,
    adjacency: Dict[str, List[Tuple[str, float]]],
    node_cost: Dict[str, float],
    edge_cost_map: Dict[Tuple[str, str], float],
    *,
    n_nodes: int,
    goal: Optional[str],
    simple_path: bool,
    exp_gamma: float,
) -> Tuple[List[str], float]:
    """
    Convert the learned tree into a concrete path by greedily following best mean reward children.
    If the tree is sparse, fall back to heuristic greedy steps.
    """
    steps_remaining = n_nodes - 1
    cur = root.current
    path = [cur]
    cost = exp_position_weight(0, exp_gamma) * node_cost.get(cur, 0.0)

    visited = set(path)

    node = root
    for step in range(steps_remaining):
        # enforce goal on last move if needed
        nbrs = [b for (b, _) in adjacency.get(cur, [])]
        if simple_path:
            nbrs = [b for b in nbrs if b not in visited]
        if goal is not None and (steps_remaining - step) == 1:
            nbrs = [b for b in nbrs if b == goal]

        if not nbrs:
            return [], float("inf")

        # Prefer using existing tree children (best mean reward)
        candidate_children = []
        if node.children:
            for b in nbrs:
                ch = node.children.get(b)
                if ch and ch.visits > 0:
                    mean = ch.total_reward / ch.visits
                    candidate_children.append((mean, b, ch))
            if candidate_children:
                candidate_children.sort(reverse=True, key=lambda t: t[0])
                _, nxt, nxt_node = candidate_children[0]
                node = nxt_node
            else:
                nxt = pick_greedy_neighbor(cur, nbrs, node_cost, edge_cost_map, pos_idx=len(path), exp_gamma=exp_gamma)
                node = node.children.get(nxt, node)  # if absent, just keep node
        else:
            nxt = pick_greedy_neighbor(cur, nbrs, node_cost, edge_cost_map, pos_idx=len(path), exp_gamma=exp_gamma)

        # apply move
        e = edge_cost_map.get((cur, nxt), float("inf"))
        pos_idx = len(path)
        cost += e + (exp_position_weight(pos_idx, exp_gamma) * node_cost.get(nxt, 0.0))
        cur = nxt
        path.append(cur)
        visited.add(cur)

    if goal is not None and path[-1] != goal:
        return [], float("inf")

    return path, cost


def pick_greedy_neighbor(
    cur: str,
    nbrs: List[str],
    node_cost: Dict[str, float],
    edge_cost_map: Dict[Tuple[str, str], float],
    *,
    pos_idx: int,
    exp_gamma: float,
) -> str:
    """Pick next step minimizing immediate incremental cost (edge + weighted node)."""
    best = None
    best_inc = float("inf")
    for b in nbrs:
        inc = edge_cost_map.get((cur, b), float("inf")) + (
            exp_position_weight(pos_idx, exp_gamma) * node_cost.get(b, 0.0)
        )
        if inc < best_inc:
            best_inc = inc
            best = b
    assert best is not None
    return best


# ----------------------------
# Main / CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to IQM metrics JSON file")
    ap.add_argument("--start", default=None, help="Start node (e.g., QB46). Required unless --sweep_starts is set.")
    ap.add_argument("--goal", default=None, help="Optional goal node (e.g., QB1). Only used when --sweep_starts is NOT set.")
    ap.add_argument("--n_nodes", type=int, default=None, help="Exact number of nodes in the path (includes start). Required unless --sweep_n is set.")

    ap.add_argument("--sweep_n", action="store_true", help="Sweep n_nodes from --n_min to --n_max (inclusive) and print the best path for each n")
    ap.add_argument("--n_min", type=int, default=2, help="Minimum n_nodes for --sweep_n")
    ap.add_argument("--n_max", type=int, default=50, help="Maximum n_nodes for --sweep_n")

    ap.add_argument("--iters", type=int, default=20000, help="Branch-and-bound DFS node-expansion budget")
    ap.add_argument("--beam_width", type=int, default=2000,
                    help="Beam width for fast incumbent search (used to seed branch-and-bound).")
    ap.add_argument("--beam_restarts", type=int, default=5,
                    help="Number of randomized beam-search restarts to find a strong incumbent quickly.")
    ap.add_argument("--c", type=float, default=1.4, help="UCB exploration constant")
    ap.add_argument("--rollout_bias", type=float, default=0.8, help="Rollout bias towards cheaper steps (0..1)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")

    ap.add_argument("--prefer_2q", default="worst", choices=["worst", "uz_cz", "irb"],
                    help="Which 2Q fidelity to use for edge costs")
    ap.add_argument("--simple_path", action="store_true", help="Disallow revisiting nodes")
    ap.add_argument("--sweep_starts", action="store_true", help="Try all possible start nodes and return the best path")
    ap.add_argument("--no_prune", action="store_true", help="Disable heuristic pruning")

    ap.add_argument("--max_keep", type=int, default=40,
                    help="If pruning enabled, keep this many best nodes by node-weight (plus mandatory nodes)")
    ap.add_argument("--drop_ge", type=float, default=None,
                    help="If pruning enabled, drop nodes with weight >= this threshold (0..1 if normalized)")

    ap.add_argument("--gamma", type=float, default=0.85,
                    help="Exponential position-weight base exp_gamma in (0,1]. Earlier nodes are weighted by exp_gamma**pos")
    ap.add_argument("--optimize_gamma", action="store_true",
                    help="Grid-search exp_gamma (log-spaced) and choose the value that minimizes the best found path cost")
    ap.add_argument("--gamma_min", type=float, default=0.55, help="Min exp_gamma for --optimize_gamma")
    ap.add_argument("--gamma_max", type=float, default=0.99, help="Max exp_gamma for --optimize_gamma")
    ap.add_argument("--gamma_steps", type=int, default=12, help="Number of exp_gamma grid points for --optimize_gamma")

    args = ap.parse_args()

    # Sweep-n validation
    if args.sweep_n:
        if args.n_min < 2:
            raise SystemExit("--n_min must be >= 2")
        if args.n_max < args.n_min:
            raise SystemExit("--n_max must be >= --n_min")
        # If the user didn't specify a start, sweep starts by default (matches desired behavior).
        if (not args.sweep_starts) and (args.start is None):
            args.sweep_starts = True
    else:
        if args.n_nodes is None:
            raise SystemExit("--n_nodes is required unless --sweep_n is set")


    # If pruning is enabled but the requested path length is large, ensure we don't prune too aggressively.
    target_max_n = args.n_max if args.sweep_n else args.n_nodes
    if (not args.no_prune) and args.simple_path and target_max_n > args.max_keep:
        args.max_keep = target_max_n

    node_m, edge_m = parse_iqm_metrics_json(args.json)
    metrics_by_node = node_metrics_to_metrics_by_node(node_m)
    node_weights = build_node_weights(metrics_by_node, w_t1=1.0, w_t2=1.0, w_ro=2.0, normalize=True)
    adjacency = build_adjacency(edge_m, prefer=args.prefer_2q)

    # Keep a copy of the full (unpruned) graph so we can automatically fall back when
    # long simple paths are requested and pruning would make them impossible.
    full_node_weights = dict(node_weights)
    full_adjacency = {k: list(v) for k, v in adjacency.items()}


    needed = args.n_max if args.sweep_n else args.n_nodes

    # Heuristic pruning, but never so aggressive that long simple paths become impossible by node count.
    if not args.no_prune:
        mandatory: List[str] = []
        if args.start:
            mandatory.append(args.start)
        if args.goal:
            mandatory.append(args.goal)
        node_weights, adjacency = prune_graph(
            node_weights, adjacency,
            max_keep=args.max_keep,
            drop_if_weight_ge=args.drop_ge,
            mandatory_keep=mandatory,
            required_min_nodes=(needed if args.simple_path else None),
        )


    # Sanity checks
    if not args.sweep_starts:
        if not args.start:
            raise SystemExit("--start is required unless --sweep_starts is set")
        if args.start not in adjacency:
            raise SystemExit(f"Start {args.start} not in adjacency after pruning. Try --no_prune or increase --max_keep.")

        if args.goal is not None and args.goal not in adjacency:
            raise SystemExit(f"Goal {args.goal} not in adjacency after pruning. Try --no_prune or increase --max_keep.")

    if not (0.0 < args.gamma <= 1.0):
        raise SystemExit("--gamma must be in (0,1]")

    # Run branch-and-bound DFS (optionally sweeping gamma and/or starts)
    def solve_for_n(n_nodes: int) -> Tuple[float, str, List[str], float, float]:
        """Solve for a single n_nodes, with automatic fallback to the full graph if pruning
        made the requested simple path length infeasible."""
        def _run(adj_use: Dict[str, List[Tuple[str, float]]], weights_use: Dict[str, float]) -> Tuple[float, str, List[str], float]:
            if args.optimize_gamma:
                return find_best_gamma_for_fixed_length(
                    adj_use,
                    weights_use,
                    start=args.start,
                    goal=args.goal,
                    n_nodes=n_nodes,
                    gamma_min=args.gamma_min,
                    gamma_max=args.gamma_max,
                    gamma_steps=args.gamma_steps,
                    sweep_starts=args.sweep_starts,
                    iterations=args.iters,
                    exploration_c=args.c,
                    rollout_bias=args.rollout_bias,
                    simple_path=args.simple_path,
                    seed=args.seed,
                    verbose=True,
                    beam_width=args.beam_width,
                    beam_restarts=args.beam_restarts,
                )
            else:
                best_gamma = float(args.gamma)
                if args.sweep_starts:
                    best_start, best_path, best_cost = find_best_start_for_fixed_length(
                        adj_use,
                        weights_use,
                        n_nodes=n_nodes,
                        exp_gamma=best_gamma,
                        iterations_per_start=args.iters,
                        exploration_c=args.c,
                        rollout_bias=args.rollout_bias,
                        simple_path=args.simple_path,
                        seed=args.seed,
                        verbose_every=10,
                        raise_on_fail=False,
                        beam_width=args.beam_width,
                        beam_restarts=args.beam_restarts,
                    )
                else:
                    best_start = args.start
                    best_path, best_cost = mcts_find_min_cost_path(
                        adj_use,
                        weights_use,
                        start=args.start,
                        goal=args.goal,
                        n_nodes=n_nodes,
                        iterations=args.iters,
                        exploration_c=args.c,
                        rollout_bias=args.rollout_bias,
                        simple_path=args.simple_path,
                        seed=args.seed,
                        exp_gamma=best_gamma,
                        beam_width=args.beam_width,
                        beam_restarts=args.beam_restarts,
                    )
                return best_gamma, best_start, best_path, best_cost

        t0_local = time.time()
        out = _run(adjacency, node_weights)

        best_gamma, best_start, best_path, best_cost = out

        # If no feasible path was found, automatically retry on the full (unpruned) graph.
        if (not best_path or not math.isfinite(best_cost)) and (not args.no_prune):
            # Only retry if pruning actually changed the graph.
            if len(adjacency) != len(full_adjacency):
                print(
                    f"[fallback] No feasible length-{n_nodes} simple path found on pruned graph; "
                    "retrying on the full (unpruned) graph..."
                )
                best_gamma, best_start, best_path, best_cost = _run(full_adjacency, full_node_weights)

        dt_local = time.time() - t0_local
        return best_gamma, best_start, best_path, best_cost, dt_local

    if args.sweep_n:
        # Sweep n_nodes in [n_min, n_max]
        for n in range(args.n_min, args.n_max + 1):
            print(f"\n===== SWEEP n_nodes={n} =====")
            best_gamma, best_start, best_path, best_cost, dt = solve_for_n(n)

            # If no feasible simple path of this length exists (or the search budget
            # was insufficient), don't crash the sweep. Just report and continue.
            if (not best_path) or (not math.isfinite(best_cost)):
                print(
                    f"No valid length-{n} path found (simple_path={args.simple_path}). "
                    "Skipping to next n."
                )
                continue

            hdr = "BEST OVER ALL START NODES" if args.sweep_starts else "BEST PATH"
            print(f"\n===== {hdr} =====")
            print(f"Time: {dt:.3f}s  iters={args.iters}  starts={len(adjacency) if args.sweep_starts else 1}")
            print(f"Exponential weight base (exp_gamma): {best_gamma:.6f}")
            print(f"Start: {best_start}")
            if args.goal and not args.sweep_starts:
                print(f"Goal:  {args.goal}")
            elif args.goal and args.sweep_starts:
                print("Goal:  (ignored with --sweep_starts)")
            print(f"Cost:  {best_cost:.6f}")
            print(f"Path length (nodes): {len(best_path)}")
            print("Path:", " -> ".join(best_path))
        return

    # Single n_nodes
    t0 = time.time()
    best_gamma, best_start, best_path, best_cost, dt = solve_for_n(args.n_nodes)

    if (not best_path) or (not math.isfinite(best_cost)):
        raise RuntimeError(
            f"No valid length-{args.n_nodes} path found (simple_path={args.simple_path}). "
            "Try increasing --iters, disabling pruning (--no_prune), or choosing a different start."
        )


    hdr = "BEST OVER ALL START NODES" if args.sweep_starts else "BEST PATH"
    print(f"\n===== {hdr} =====")
    print(f"Time: {dt:.3f}s  iters={args.iters}  starts={len(adjacency) if args.sweep_starts else 1}")
    print(f"Exponential weight base (exp_gamma): {best_gamma:.6f}")
    print(f"Start: {best_start}")
    if args.goal and not args.sweep_starts:
        print(f"Goal:  {args.goal}")
    elif args.goal and args.sweep_starts:
        print("Goal:  (ignored with --sweep_starts)")
    print(f"Cost:  {best_cost:.6f}")
    print(f"Path length (nodes): {len(best_path)}")
    print("Path:", " -> ".join(best_path))



if __name__ == "__main__":
    main()