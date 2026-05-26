import numpy as np
import networkx as nx
import igraph as ig
from joblib import Parallel, delayed
from math import ceil
import multiprocessing
from itertools import product
from collections import defaultdict
from networkx.algorithms.isomorphism import categorical_node_match, numerical_edge_match

def enumerate_contractions_memory_efficient_parallel(N, n_jobs=-1):
    """
    Parallel, memory-efficient enumeration of inequivalent weighted graphs
    representing contractions of N copies (N even) of an antisymmetric 3-form H.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")
    if N < 2:
        return [], []

    # ——————————————————————————————————————————————
    # Helpers
    def create_igraph(sol, num_nodes):
        G = ig.Graph(n=num_nodes, directed=False)
        edges, weights = [], []
        for (u, v), w in sorted(sol.items()):
            if w > 0:
                edges.append((u, v))
                weights.append(w)
        if edges:
            G.add_edges(edges)
            G.es['weight'] = weights
            max_idx = max(max(u, v) for u, v in edges)
            if max_idx < num_nodes - 1:
                G.add_vertices(range(max_idx + 1, num_nodes))
        return G

    def graph_invariant(G):
        # 1) weighted degree (strength) sequence
        degs = tuple(sorted(G.strength(weights='weight')))
        # 2) eigenvalues of weighted adj matrix (symmetric solver)
        A = np.zeros((N, N))
        for e in G.es:
            u, v = e.tuple
            w = e['weight']
            A[u, v] = A[v, u] = w
        eigs = tuple(np.sort(np.round(np.linalg.eigvalsh(A), 5)).tolist())
        return (degs, eigs)

    def generate_choices(start, end, rem, degs):
        if rem == 0:
            yield {}
            return
        if start >= end:
            return
        if degs[start] == 0:
            yield from generate_choices(start + 1, end, rem, degs)
            return
        max_m = min(degs[start], rem)
        future_sum = sum(degs[k] for k in range(start + 1, end))
        for m in range(max_m + 1):
            needed = rem - m
            if needed > future_sum:
                continue
            for tail in generate_choices(start + 1, end, needed, degs):
                d = {}
                if m > 0:
                    d[start] = m
                d.update(tail)
                yield d

    def recurse_stream(v, degs, graph_dict, patterns):
        while v < N and degs[v] == 0:
            v += 1
        if v == N:
            yield graph_dict.copy()
            return

        need = degs[v]
        for choice in generate_choices(v + 1, N, need, degs):
            deg_changes, graph_changes = [], []
            ok = True
            for j, m in choice.items():
                if degs[j] < m:
                    ok = False
                    break
                degs[j] -= m
                deg_changes.append((j, m))
                graph_dict[(v, j)] = graph_dict.get((v, j), 0) + m
                graph_changes.append(((v, j), m))
            if not ok:
                for j, m in deg_changes:
                    degs[j] += m
                continue

            pat = tuple(sorted(choice.items()))
            if v > 0 and patterns[v - 1] is not None and pat < patterns[v - 1]:
                for j, m in deg_changes:
                    degs[j] += m
                for edge, m in graph_changes:
                    graph_dict[edge] -= m
                    if graph_dict[edge] == 0:
                        del graph_dict[edge]
                continue

            old_pat, old_deg_v = patterns[v], degs[v]
            patterns[v], degs[v] = pat, 0
            yield from recurse_stream(v + 1, degs, graph_dict, patterns)
            patterns[v], degs[v] = old_pat, old_deg_v
            for j, m in deg_changes:
                degs[j] += m
            for edge, m in graph_changes:
                graph_dict[edge] -= m
                if graph_dict[edge] == 0:
                    del graph_dict[edge]

    # ——————————————————————————————————————————————
    # 1) Generate all vertex-0 choices once
    init_deg = [3] * N
    first_choices = list(generate_choices(1, N, init_deg[0], init_deg))

    # 2) Split into chunks
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    chunk_size = ceil(len(first_choices) / n_jobs)
    chunks = [
        first_choices[i : i + chunk_size]
        for i in range(0, len(first_choices), chunk_size)
    ]

    # 3) Worker that processes ONE chunk, streaming & locally dedup’ing
    def process_chunk(chunk_choices):
        local_reps = {}   # inv → [igraph reps]
        local_sols = []   # raw dicts
        for choice0 in chunk_choices:
            degs = init_deg.copy()
            graph_dict = {}
            patterns = [None] * N
            for j, m in choice0.items():
                degs[j] -= m
                graph_dict[(0, j)] = m
            degs[0] = 0
            patterns[0] = tuple(sorted(choice0.items()))

            for sol in recurse_stream(1, degs, graph_dict, patterns):
                G_ig = create_igraph(sol, N)
                inv  = graph_invariant(G_ig)

                bucket = local_reps.get(inv, [])
                is_new = True
                for rep in bucket:
                    if G_ig.isomorphic_vf2(
                            rep,
                            edge_color1=G_ig.es['weight'],
                            edge_color2=rep.es['weight']
                    ):
                        is_new = False
                        break
                if not is_new:
                    continue

                bucket.append(G_ig)
                local_reps[inv] = bucket
                local_sols.append(sol)

        return local_sols

    # 4) Parallel map
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk) for chunk in chunks
    )

    # 5) Flatten & FINAL dedup across chunks
    combined = [sol for sub in results for sol in sub]
    reps = {}
    final_sols = []
    for sol in combined:
        G_ig = create_igraph(sol, N)
        if not G_ig.is_connected():
            continue
        inv  = graph_invariant(G_ig)

        bucket = reps.get(inv, [])
        is_new = True
        for rep in bucket:
            if G_ig.isomorphic_vf2(
                    rep,
                    edge_color1=G_ig.es['weight'],
                    edge_color2=rep.es['weight']
            ):
                is_new = False
                break
        if not is_new:
            continue

        bucket.append(G_ig)
        reps[inv] = bucket
        final_sols.append(sol)

    # 6) Build the networkx graphs
    final_graphs = []
    for sol in final_sols:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for (i, j), w in sol.items():
            if w > 0:
                G.add_edge(i, j, weight=w)
        final_graphs.append(G)

    return final_sols, final_graphs


def get_colored_graphs(n, n_jobs=-1):
    """
    Enumerate all non-isomorphic, connected, 2-colored contractions on n nodes.
    Optimized: only compare colorings of the same base graph and same red count.
    """
    _, base_graphs = enumerate_contractions_memory_efficient_parallel(n, n_jobs)
    connected = [G for G in base_graphs if nx.is_connected(G)]

    node_match = categorical_node_match('color', None)
    edge_match = numerical_edge_match('weight', 1)

    unique_colored = []

    for base in connected:
        # inner buckets: red_count → list of seen colorings of this base graph
        red_buckets = defaultdict(list)

        for coloring in product([0, 1], repeat=n):
            red_count = sum(coloring)
            Gc = base.copy()
            for i, c in enumerate(coloring):
                Gc.nodes[i]['color'] = c

            # compare only within this base graph + same red count
            bucket = red_buckets[red_count]
            is_new = True
            for other in bucket:
                if nx.is_isomorphic(Gc, other, node_match=node_match, edge_match=edge_match):
                    is_new = False
                    break
            if is_new:
                bucket.append(Gc)

        for lst in red_buckets.values():
            unique_colored.extend(lst)

    return unique_colored
