"""Small dependency-free genetic optimizer for continuous multi-objective search.

The public entry point, :func:`ga_minimize_pareto`, implements a compact
NSGA-II style loop using only numpy.  Callers may provide a batch evaluator
when the expensive simulation backend can isolate each run in its own
workspace.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Record = Dict[str, Any]
Individual = Dict[str, Any]


def _finite_objectives(record: Record) -> Optional[np.ndarray]:
    vals = record.get("objective_values")
    if vals is None:
        vals = [record.get("objective_value", float("inf"))]
    try:
        arr = np.asarray(vals, dtype=float).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if record.get("relap_failed") or record.get("post_run_error"):
        return None
    return arr


def _objectives_for_sort(ind: Individual, objective_dim: int) -> np.ndarray:
    arr = ind.get("objectives")
    if arr is None:
        return np.full(objective_dim, float("inf"), dtype=float)
    arr = np.asarray(arr, dtype=float).flatten()
    if arr.size == objective_dim:
        return arr
    out = np.full(objective_dim, float("inf"), dtype=float)
    n = min(arr.size, objective_dim)
    if n > 0:
        out[:n] = arr[:n]
    return out


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when objective vector ``a`` Pareto-dominates ``b``."""
    if not np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
        return False
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_front_indices(
    individuals: Sequence[Individual],
    objective_dim: int,
) -> List[List[int]]:
    n = len(individuals)
    if n == 0:
        return []

    objs = [_objectives_for_sort(ind, objective_dim) for ind in individuals]
    dominates: List[List[int]] = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                dominates[p].append(q)
            elif _dominates(objs[q], objs[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def _crowding_distance(
    individuals: Sequence[Individual],
    front: Sequence[int],
    objective_dim: int,
) -> Dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distance = {idx: 0.0 for idx in front}
    objs = {
        idx: _objectives_for_sort(individuals[idx], objective_dim)
        for idx in front
    }
    for k in range(objective_dim):
        ordered = sorted(front, key=lambda idx: objs[idx][k])
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        lo = objs[ordered[0]][k]
        hi = objs[ordered[-1]][k]
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        for j in range(1, len(ordered) - 1):
            prev_v = objs[ordered[j - 1]][k]
            next_v = objs[ordered[j + 1]][k]
            distance[ordered[j]] += float((next_v - prev_v) / (hi - lo))
    return distance


def _assign_rank_and_crowding(
    individuals: Sequence[Individual],
    objective_dim: int,
) -> List[List[int]]:
    fronts = _non_dominated_front_indices(individuals, objective_dim)
    for rank, front in enumerate(fronts):
        cd = _crowding_distance(individuals, front, objective_dim)
        for idx in front:
            individuals[idx]["rank"] = rank
            individuals[idx]["crowding"] = cd.get(idx, 0.0)
    return fronts


def _weighted_score(record: Record) -> float:
    try:
        return float(record.get("objective_value", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _select_population(
    candidates: Sequence[Individual],
    size: int,
    objective_dim: int,
) -> List[Individual]:
    fronts = _assign_rank_and_crowding(candidates, objective_dim)
    selected: List[Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(candidates[idx] for idx in front)
            continue
        ordered = sorted(
            front,
            key=lambda idx: (
                -float(candidates[idx].get("crowding", 0.0)),
                _weighted_score(candidates[idx]["record"]),
            ),
        )
        selected.extend(candidates[idx] for idx in ordered[: size - len(selected)])
        break
    return list(selected)


def _pareto_front_records(
    individuals: Sequence[Individual],
    objective_dim: int,
) -> List[Record]:
    valid = [ind for ind in individuals if ind.get("objectives") is not None]
    if not valid:
        return []
    fronts = _non_dominated_front_indices(valid, objective_dim)
    if not fronts:
        return []
    front_records = [dict(valid[idx]["record"]) for idx in fronts[0]]
    front_records.sort(
        key=lambda rec: (
            _weighted_score(rec),
            int(rec.get("evaluation", rec.get("iteration", 10**9)) or 10**9),
        )
    )
    return front_records


def _tournament(
    population: Sequence[Individual],
    rng: np.random.Generator,
) -> Individual:
    i, j = rng.integers(0, len(population), size=2)
    a = population[int(i)]
    b = population[int(j)]
    key_a = (
        int(a.get("rank", 10**9)),
        -float(a.get("crowding", 0.0)),
        _weighted_score(a["record"]),
    )
    key_b = (
        int(b.get("rank", 10**9)),
        -float(b.get("crowding", 0.0)),
        _weighted_score(b["record"]),
    )
    return a if key_a <= key_b else b


def _candidate_from_record(record: Record, d: int) -> Optional[np.ndarray]:
    vals = record.get("parameter_values")
    if vals is None and "parameter_value" in record:
        vals = [record["parameter_value"]]
    try:
        arr = np.asarray(vals, dtype=float).flatten()
    except (TypeError, ValueError):
        return None
    if arr.shape[0] != d or not np.all(np.isfinite(arr)):
        return None
    return arr


def ga_minimize_pareto(
    evaluate: Callable[[np.ndarray], Record],
    bounds: Sequence[Tuple[float, float]],
    *,
    max_evaluations: int,
    warm_start_records: Optional[Sequence[Record]] = None,
    population_size: Optional[int] = None,
    random_seed: int = 12345,
    crossover_probability: float = 0.9,
    mutation_scale: float = 0.12,
    stop_requested: Optional[Callable[[], bool]] = None,
    evaluate_many: Optional[Callable[[Sequence[np.ndarray]], Sequence[Record]]] = None,
    parallel_workers: int = 1,
) -> Dict[str, Any]:
    """Minimize multiple objectives over continuous bounds using a small NSGA-II loop."""
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.ndim != 2 or bounds_arr.shape[1] != 2:
        raise ValueError(f"bounds must be List[(a,b)], got shape {bounds_arr.shape}")
    if bounds_arr.shape[0] == 0:
        raise ValueError("bounds must not be empty")
    a_vec = bounds_arr[:, 0]
    b_vec = bounds_arr[:, 1]
    if np.any(a_vec >= b_vec):
        raise ValueError(f"Each bound (a,b) must have a<b, got {bounds!r}")
    d = bounds_arr.shape[0]

    max_evaluations = max(0, int(max_evaluations))
    if max_evaluations == 0:
        return {
            "success": False,
            "stop_reason": "max_evaluations_zero",
            "evaluations": [],
            "pareto_front": [],
            "best_record": None,
            "population_size": 0,
        }

    if population_size is None:
        pop_size = min(max_evaluations, max(8, 2 * d + 4))
    else:
        pop_size = min(max_evaluations, max(2, int(population_size)))
    batch_size = max(1, int(parallel_workers))

    rng = np.random.default_rng(seed=int(random_seed))
    crossover_probability = min(1.0, max(0.0, float(crossover_probability)))
    mutation_scale = max(0.0, float(mutation_scale))

    evaluations: List[Individual] = []
    population: List[Individual] = []
    objective_dim = 1

    def add_record(record: Record, x: np.ndarray) -> Individual:
        nonlocal objective_dim
        rec = dict(record)
        rec.setdefault("parameter_values", x.tolist())
        if "parameter_value" not in rec and x.size:
            rec["parameter_value"] = float(x[0])
        obj = _finite_objectives(rec)
        if obj is not None:
            objective_dim = max(objective_dim, int(obj.size))
        ind = {
            "x": np.array(x, dtype=float),
            "record": rec,
            "objectives": obj,
            "rank": 10**9,
            "crowding": 0.0,
        }
        evaluations.append(ind)
        return ind

    def eval_candidate(x: np.ndarray) -> Optional[Individual]:
        if stop_requested is not None and stop_requested():
            return None
        x = np.maximum(a_vec, np.minimum(b_vec, np.asarray(x, dtype=float)))
        record = evaluate(x.copy())
        return add_record(record, x)

    def eval_candidates(candidates: Sequence[np.ndarray]) -> List[Individual]:
        if stop_requested is not None and stop_requested():
            return []
        clamped = [
            np.maximum(a_vec, np.minimum(b_vec, np.asarray(x, dtype=float))).copy()
            for x in candidates
        ]
        if not clamped:
            return []

        if evaluate_many is not None and len(clamped) > 1:
            records = list(evaluate_many([x.copy() for x in clamped]))
            if len(records) != len(clamped):
                raise ValueError(
                    "evaluate_many must return one record for each candidate "
                    f"({len(records)} vs {len(clamped)})"
                )
            return [add_record(record, x) for record, x in zip(records, clamped)]

        out: List[Individual] = []
        for x in clamped:
            ind = eval_candidate(x)
            if ind is None:
                break
            out.append(ind)
        return out

    if warm_start_records:
        for record in warm_start_records:
            if len(evaluations) >= max_evaluations:
                break
            x = _candidate_from_record(record, d)
            if x is None:
                continue
            x = np.maximum(a_vec, np.minimum(b_vec, x))
            population.append(add_record(record, x))

    seeds: List[np.ndarray] = [
        (a_vec + b_vec) / 2.0,
        a_vec.copy(),
        b_vec.copy(),
    ]
    for axis in range(d):
        lo_mid = (a_vec + b_vec) / 2.0
        hi_mid = lo_mid.copy()
        lo_mid[axis] = a_vec[axis]
        hi_mid[axis] = b_vec[axis]
        seeds.extend([lo_mid, hi_mid])

    seed_i = 0
    while len(population) < pop_size and len(evaluations) < max_evaluations:
        if stop_requested is not None and stop_requested():
            break
        pending: List[np.ndarray] = []
        while (
            len(population) + len(pending) < pop_size
            and len(evaluations) + len(pending) < max_evaluations
            and len(pending) < batch_size
        ):
            if seed_i < len(seeds):
                cand = seeds[seed_i]
                seed_i += 1
            else:
                cand = a_vec + rng.random(d) * (b_vec - a_vec)
            pending.append(cand)
        if not pending:
            break
        inds = eval_candidates(pending)
        if not inds:
            break
        population.extend(inds)

    if not population:
        return {
            "success": False,
            "stop_reason": "stopped_by_user",
            "evaluations": [ind["record"] for ind in evaluations],
            "pareto_front": [],
            "best_record": None,
            "population_size": pop_size,
        }

    stop_reason = "max_evaluations_reached"
    while len(evaluations) < max_evaluations:
        if stop_requested is not None and stop_requested():
            stop_reason = "stopped_by_user"
            break

        _assign_rank_and_crowding(population, objective_dim)
        children: List[Individual] = []
        remaining = max_evaluations - len(evaluations)
        child_budget = min(pop_size, remaining)

        child_candidates: List[np.ndarray] = []
        for _ in range(child_budget):
            p1 = _tournament(population, rng)
            p2 = _tournament(population, rng)
            if rng.random() < crossover_probability:
                alpha = rng.random(d)
                child_x = alpha * p1["x"] + (1.0 - alpha) * p2["x"]
            else:
                child_x = p1["x"].copy()

            if mutation_scale > 0.0:
                mask = rng.random(d) < max(1.0 / d, 0.2)
                if not np.any(mask):
                    mask[int(rng.integers(0, d))] = True
                sigma = mutation_scale * (b_vec - a_vec)
                child_x = child_x.copy()
                child_x[mask] += rng.normal(0.0, sigma[mask])
            child_x = np.maximum(a_vec, np.minimum(b_vec, child_x))
            child_candidates.append(child_x)

        for start in range(0, len(child_candidates), batch_size):
            inds = eval_candidates(child_candidates[start : start + batch_size])
            if not inds:
                stop_reason = "stopped_by_user"
                break
            children.extend(inds)

        population = _select_population(
            list(population) + children,
            pop_size,
            objective_dim,
        )

        if stop_reason == "stopped_by_user":
            break

    pareto_front = _pareto_front_records(evaluations, objective_dim)
    best_record = min(pareto_front, key=_weighted_score) if pareto_front else None
    return {
        "success": best_record is not None,
        "stop_reason": stop_reason if best_record is not None else "all_evaluations_failed",
        "evaluations": [ind["record"] for ind in evaluations],
        "pareto_front": pareto_front,
        "best_record": dict(best_record) if best_record is not None else None,
        "population_size": pop_size,
    }
