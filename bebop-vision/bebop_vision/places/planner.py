"""Multi-leg planning over the place graph.

Nodes are places; a directed edge is usable only when it carries at least
one taught route (a trail the follower can drive).  Edge cost is the
expected time: the edge's median successful duration when it has one,
otherwise the newest route's length at a conservative speed; unreliable
edges are penalised by their failure rate.  Plain Dijkstra — the graph is a
handful of rooms, not a city.

Pure functions over dicts so the planner can be tested without a store.
"""
from __future__ import annotations

import heapq
import math

ASSUMED_SPEED_M_S = 0.2
UNKNOWN_EDGE_COST_S = 60.0


def edge_cost(edge: dict, routes: dict) -> float | None:
    """Seconds to expect for one edge, or None when it cannot be followed.

    `routes` maps route_id -> route dict (needs `length_m`, `created_at`).
    """
    ids = [r for r in edge.get("route_ids") or [] if r in routes]
    if not ids:
        return None
    newest = max(ids, key=lambda r: routes[r].get("created_at", 0.0))
    base = edge.get("median_duration_s")
    if not base or base <= 0:
        length = float(routes[newest].get("length_m") or 0.0)
        base = (length / ASSUMED_SPEED_M_S) if length > 0 \
            else UNKNOWN_EDGE_COST_S
    trav = int(edge.get("traversals") or 0)
    succ = int(edge.get("successes") or 0)
    rate = (succ / trav) if trav > 0 else 1.0
    return float(base) * (1.0 + 2.0 * (1.0 - rate))


def plan(graph: dict, routes: dict, from_place: str, to_place: str,
         *, min_success_rate: float = 0.0) -> list[dict]:
    """Cheapest list of legs [{from, to, route_id, cost_s}] or [] if none.

    `graph` is `PlaceStore.graph()`; `routes` maps route_id -> route dict.
    Edges whose success rate is below `min_success_rate` (and that have
    been tried) are skipped.
    """
    if from_place == to_place:
        return []
    adj: dict[str, list[tuple[str, str, float]]] = {}
    for e in graph.get("edges", []):
        cost = edge_cost(e, routes)
        if cost is None:
            continue
        trav = int(e.get("traversals") or 0)
        if trav > 0 and (int(e.get("successes") or 0) / trav) < min_success_rate:
            continue
        ids = [r for r in e["route_ids"] if r in routes]
        newest = max(ids, key=lambda r: routes[r].get("created_at", 0.0))
        adj.setdefault(e["from"], []).append((e["to"], newest, cost))
    best = {from_place: 0.0}
    prev: dict[str, tuple[str, str, float]] = {}
    heap = [(0.0, from_place)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > best.get(u, math.inf):
            continue
        if u == to_place:
            break
        for v, rid, w in adj.get(u, []):
            nd = d + w
            if nd < best.get(v, math.inf):
                best[v] = nd
                prev[v] = (u, rid, w)
                heapq.heappush(heap, (nd, v))
    if to_place not in best:
        return []
    legs = []
    v = to_place
    while v != from_place:
        u, rid, w = prev[v]
        legs.append({"from": u, "to": v, "route_id": rid,
                     "cost_s": round(w, 1)})
        v = u
    legs.reverse()
    return legs
