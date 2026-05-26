import numpy as np
import string, itertools
from sympy.combinatorics.named_groups import SymmetricGroup # Not used
import networkx as nx
import networkx.algorithms.isomorphism as iso # Replaced by igraph
from collections import Counter, defaultdict # Use defaultdict for graph
from joblib import Parallel, delayed
import igraph as ig
import time # For basic profiling if needed

def get_weights(n_nodes, weight_dict):
    weights = np.zeros(n_nodes)
    for key, value in weight_dict.items():
        weights[key[0]] += value
        weights[key[1]] += value
    
    return weights

def enumerate_contractions_by_nodes_symmetry_parallel_optimized(N, n_jobs=-1):
    """
    Optimized enumeration of inequivalent weighted graphs representing contractions
    of N copies (N even) of an antisymmetric 3-form H.
    Parallelizes over the assignments for vertex 0, uses in-place updates,
    and optimized deduplication with igraph.

    Returns:
      final_solutions: a list of dictionaries mapping (i, j) with i < j to the weight on that edge.
      final_graphs: the corresponding list of NetworkX graphs (one per inequivalent contraction).
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")
    if N < 2:
         return [], [] # Or handle as appropriate for N=0

    initial_deg = [3] * N  # Use a list, as in-place modification is easy
    patterns = [None] * N  # To record canonical pattern for each vertex

    # --- Helper: igraph Conversion (Optimized) ---
    # Create igraph object directly from edge list and weights
    def create_igraph_from_solution(sol_dict, num_nodes):
        ig_G = ig.Graph(n=num_nodes, directed=False)
        edges = []
        weights = []
        # Ensure keys are sorted (i, j) with i < j for consistency
        # The recursive construction naturally produces keys (v, j) where v < j
        sorted_items = sorted(sol_dict.items()) 
        for (u, v), w in sorted_items:
            if w > 0:
                edges.append((u, v))
                weights.append(w)
        
        if not edges: # Handle disconnected graphs (all nodes isolated)
             return ig_G # Return graph with nodes but no edges

        try:
            ig_G.add_edges(edges)
            ig_G.es['weight'] = weights
            # Add isolated nodes if any node index wasn't in an edge
            max_node_in_edges = -1
            if edges:
                max_node_in_edges = max(max(u,v) for u,v in edges)
            if max_node_in_edges < num_nodes - 1:
                 # igraph automatically adds vertices up to the max index in add_edges
                 # We need to explicitly add any higher-indexed isolated vertices
                 ig_G.add_vertices(range(max_node_in_edges + 1, num_nodes))

        except Exception as e:
             print(f"Error creating igraph: {e}")
             print(f"Nodes: {num_nodes}, Edges: {edges}, Weights: {weights}")
             # Depending on igraph version, adding edges might fail if nodes aren't pre-declared
             # The ig.Graph(n=num_nodes) should handle this. Re-check logic if errors persist.
             # Let's try adding vertices first explicitly
             ig_G_alt = ig.Graph(directed=False)
             ig_G_alt.add_vertices(num_nodes)
             if edges:
                 ig_G_alt.add_edges(edges)
                 ig_G_alt.es['weight'] = weights
             return ig_G_alt


        return ig_G

    # --- Helper: Stronger Graph Invariant ---
    # Uses igraph directly for speed if possible, or converts from solution dict
    def graph_invariant_igraph(ig_G):
        try:
            # 1. Weighted Degree Sequence (sorted)
            degrees = tuple(sorted(ig_G.strength(weights='weight'))) # igraph strength = weighted degree

            # 2. Eigenvalue Spectrum of Weighted Adjacency Matrix (rounded)
            # Getting weighted adjacency matrix from igraph can be tricky/version-dependent
            # Let's build it manually for robustness or use NetworkX temporarily if needed
            # Manual construction:
            adj_matrix = np.zeros((N, N))
            for edge in ig_G.es:
                u, v = edge.tuple
                w = edge['weight']
                adj_matrix[u, v] = w
                adj_matrix[v, u] = w # Undirected

            eigvals = np.sort(np.round(np.linalg.eigvals(adj_matrix), decimals=5))
            
            return (degrees, tuple(eigvals.tolist()))
        except Exception as e:
            # Fallback or error handling
            print(f"Warning: Could not compute invariant for igraph {ig_G.summary()}: {e}")
            # As a basic fallback, maybe return hash of edge/weight tuples
            # Note: this fallback is WEAK and might group non-isomorphic graphs
            edge_tuples = tuple(sorted([(e.source, e.target, e['weight']) for e in ig_G.es]))
            return (hash(edge_tuples),) # Very basic fallback


    # --- Recursive Choice Generator (Minor optimization: yield from) ---
    def generate_choices(start, end, remaining, current_deg):
        """
        Recursively generate assignments for vertices in range(start, end).
        Yields dictionaries mapping vertex j to m (only for m > 0).
        Uses yield from.
        """
        if remaining == 0:
            yield {}
            return
        if start >= end:
            return

        # Check if node 'start' has enough degree *before* iterating
        if current_deg[start] == 0:
             yield from generate_choices(start + 1, end, remaining, current_deg)
             return

        max_m = min(current_deg[start], remaining)
        for m in range(max_m + 1):
            # Optimization: Check if remaining degree is sufficient *before* recursing deep
            needed_for_rest = remaining - m
            available_in_rest = sum(current_deg[k] for k in range(start + 1, end))
            if needed_for_rest > available_in_rest:
                 continue # Pruning: Not enough degree downstream

            # Recurse
            for rest in generate_choices(start + 1, end, remaining - m, current_deg):
                d = {}
                if m > 0:
                    d[start] = m
                d.update(rest)
                yield d


    # --- Optimized Recursive Function (In-place updates & Backtracking) ---
    all_solutions_list = [] # Store raw solution dictionaries

    def rec_optimized(v, current_deg, current_graph, current_patterns):
        """
        Recursive function using in-place updates and backtracking. Appends valid solutions
        to the global `all_solutions_list`.
        """
        # Skip vertices that are already saturated.
        while v < N and current_deg[v] == 0:
            v += 1

        if v == N:
            # Base case: Found a potential solution. Add a copy to the list.
            # Ensure all degrees are zero
            if all(d == 0 for d in current_deg):
                 # Create a copy of the current_graph dictionary for storage
                 all_solutions_list.append(current_graph.copy())
            # else: # This case should ideally not be reached if logic is correct
                 # print("Warning: Reached end with non-zero degrees:", current_deg)
            return

        need = current_deg[v] # Remaining half-edges for vertex v

        # --- Iterate through choices using the generator ---
        for choice in generate_choices(v + 1, N, need, current_deg):
            # --- Try applying the choice ---
            # Store changes to backtrack
            deg_changes = {} # Store {j: m} for degree changes
            graph_changes = {} # Store {(v, j): m} for graph changes
            valid = True

            for j, m in choice.items():
                if current_deg[j] < m : # Check if sufficient degree exists
                     valid = False
                     break # Invalid choice, cannot subtract m
                
                # Update degree (in-place)
                current_deg[j] -= m
                deg_changes[j] = m # Record change

                # Update graph (in-place using defaultdict)
                key = (v, j) # v < j is guaranteed by generate_choices start
                current_graph[key] = current_graph.get(key, 0) + m # Use get for defaultdict pattern
                graph_changes[key] = m # Record change amount for this specific step
            
            if not valid:
                 # Backtrack degree changes made *within this choice iteration* before breaking
                 for j_back, m_back in deg_changes.items():
                     current_deg[j_back] += m_back
                 # No graph changes to backtrack here as they didn't happen for the invalid part
                 continue # Try next choice

            # --- Choice application successful so far ---

            # Calculate local pattern for symmetry check
            local_pattern = tuple(sorted(choice.items())) # Pattern based on assigned partners {j:m}

            # Symmetry breaking: if v > 0, require pattern >= previous vertex's pattern
            symmetry_ok = True
            if v > 0 and current_patterns[v - 1] is not None and local_pattern < current_patterns[v - 1]:
                 symmetry_ok = False

            if symmetry_ok:
                # Store old pattern, set new, and recurse
                old_pattern = current_patterns[v]
                current_patterns[v] = local_pattern
                current_deg[v] = 0 # Vertex v is now saturated

                rec_optimized(v + 1, current_deg, current_graph, current_patterns)

                # Backtrack state AFTER recursion returns
                current_patterns[v] = old_pattern # Restore pattern
                current_deg[v] = need # Restore degree of v

            # --- Backtrack changes from this choice regardless of symmetry check outcome ---
            # Backtrack degree changes
            for j_back, m_back in deg_changes.items():
                current_deg[j_back] += m_back
            # Backtrack graph changes
            for key_back, m_added in graph_changes.items():
                current_graph[key_back] -= m_added
                if current_graph[key_back] == 0:
                    del current_graph[key_back] # Clean up zero-weight edges

        # End of loop for choices for vertex v


    # --- Parallel Processing Setup ---
    # Vertex 0 processing needs careful setup for parallel execution
    
    # Generate choices for vertex 0 *once*
    first_vertex_choices = list(generate_choices(1, N, initial_deg[0], initial_deg))

    # Clear the global list before parallel processing
    all_solutions_list = [] 
    
    # Define the worker function for Joblib
    # It needs to manage its own state copies or work on independent segments
    # We will pass copies of initial state to each worker to ensure independence
    def process_choice_parallel(choice, initial_deg_copy, N_nodes):
        # Create local state for this parallel branch
        local_deg = initial_deg_copy.copy()
        local_graph = {} # Use standard dict here, or defaultdict(int)
        local_patterns = [None] * N_nodes # Each branch needs its own pattern history
        
        # Apply the choice for vertex 0 to the local state
        valid = True
        local_pattern_list = []
        for j, m in choice.items():
            if local_deg[j] < m:
                valid = False
                break
            local_deg[j] -= m
            if m > 0:
                 local_graph[(0, j)] = m # Key (0, j) since 0 < j
                 local_pattern_list.append((j, m)) # Use list for sorting

        if not valid:
            return [] # Return empty list for invalid branches

        local_deg[0] = 0 # Saturate vertex 0
        local_patterns[0] = tuple(sorted(local_pattern_list)) # Set pattern for v=0

        # Use a temporary list within the worker to collect results
        worker_solutions = []
        
        # Define the recursive function *inside* the worker or pass the list
        # Option 1: Define inner recursive function using worker_solutions
        def rec_worker(v, current_deg, current_graph, current_patterns):
            while v < N_nodes and current_deg[v] == 0: v += 1
            if v == N_nodes:
                if all(d == 0 for d in current_deg):
                    worker_solutions.append(current_graph.copy())
                return

            need = current_deg[v]
            for ch in generate_choices(v + 1, N_nodes, need, current_deg):
                deg_changes, graph_changes = {}, {}
                choice_valid = True
                for j, m in ch.items():
                    if current_deg[j] < m: choice_valid = False; break
                    current_deg[j] -= m
                    deg_changes[j] = m
                    key = (v, j)
                    current_graph[key] = current_graph.get(key, 0) + m
                    graph_changes[key] = m
                
                if not choice_valid:
                     for j_back, m_back in deg_changes.items(): current_deg[j_back] += m_back
                     continue

                local_pat = tuple(sorted(ch.items()))
                symm_ok = True
                if v > 0 and current_patterns[v - 1] is not None and local_pat < current_patterns[v - 1]:
                     symm_ok = False

                if symm_ok:
                    old_pat = current_patterns[v]
                    current_patterns[v] = local_pat
                    current_deg[v] = 0
                    rec_worker(v + 1, current_deg, current_graph, current_patterns)
                    current_patterns[v] = old_pat
                    current_deg[v] = need

                for j_back, m_back in deg_changes.items(): current_deg[j_back] += m_back
                for key_back, m_added in graph_changes.items():
                    current_graph[key_back] -= m_added
                    if current_graph[key_back] == 0: del current_graph[key_back]
        
        # Start the recursion for this worker from vertex 1
        rec_worker(1, local_deg, local_graph, local_patterns)
        
        return worker_solutions

    # --- Run Parallel Computation ---
    # print(f"Processing {len(first_vertex_choices)} choices for vertex 0 in parallel...")
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(process_choice_parallel)(choice, initial_deg, N) 
        for choice in first_vertex_choices
    )

    # --- Merge results from parallel runs ---
    # The global all_solutions_list is not used now, results are in parallel_results
    merged_solutions = [sol for branch_result in parallel_results for sol in branch_result]
    # print(f"Generated {len(merged_solutions)} raw solutions.")

    if not merged_solutions:
        return [], []

    # --- Deduplication using igraph (Optimized) ---
    # print("Starting deduplication...")
    start_dedup_time = time.time()

    # 1. Create all igraph graphs ONCE
    # print("Creating igraph objects...")
    igraphs = [create_igraph_from_solution(sol, N) for sol in merged_solutions]
    
    # 2. Compute invariants for all igraph graphs
    # print("Calculating invariants...")
    invariants = [graph_invariant_igraph(ig_G) for ig_G in igraphs]

    # 3. Deduplicate based on invariants and isomorphism checks
    # print("Checking isomorphism...")
    representatives = {}  # Map: invariant -> list of representative igraph objects
    unique_indices = []   # Stores indices of unique solutions in merged_solutions

    for i, ig_G in enumerate(igraphs):
        inv = invariants[i]
        bucket = representatives.get(inv)
        is_unique = True

        if bucket: # Check against representatives in the same bucket
             # Use igraph's isomorphism check with weights
            if any(ig_G.isomorphic_vf2(rep_ig, 
                                        edge_color1=ig_G.es['weight'] if ig_G.ecount() > 0 else None, 
                                        edge_color2=rep_ig.es['weight'] if rep_ig.ecount() > 0 else None)
                   for rep_ig in bucket):
                is_unique = False
        
        if is_unique:
            unique_indices.append(i)
            if bucket is None:
                representatives[inv] = [ig_G]
            else:
                bucket.append(ig_G) # Add as a new representative for this invariant

    dedup_time = time.time() - start_dedup_time
    # print(f"Deduplication finished in {dedup_time:.2f}s. Found {len(unique_indices)} unique graphs.")

    # 4. Build final results using the unique indices
    final_solutions = [merged_solutions[i] for i in unique_indices]
    
    # Create NetworkX graphs only for the unique solutions
    final_graphs = []
    for idx in unique_indices:
        sol = merged_solutions[idx]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for (i, j), w in sol.items():
             if w > 0: # Should always be true based on how graph dict is built
                 G.add_edge(i, j, weight=w)
        final_graphs.append(G)

    return final_solutions, final_graphs


def get_inequivalent_graphs(n):
    solutions, graphs = enumerate_contractions_by_nodes_symmetry_parallel_optimized(n)
            
    return [g for g in graphs if nx.is_connected(g)]

def contract_strings(str_list, t1, t2, free_letters):
    this_letter = free_letters[0]
    new_free_letters = free_letters[1:]

    str_list[t1] = str_list[t1].replace('?', this_letter, 1)
    str_list[t2] = str_list[t2].replace('?', this_letter, 1)

    return str_list, new_free_letters

def graph_to_einsum(graph):
    free_letters = string.ascii_letters
    nodes = graph.nodes
    n_nodes = len(nodes)
    n_indices = graph.degree(list(nodes)[0], weight='weight')
    
    letters_list = [ '?'*n_indices ]*n_nodes

    for e in graph.edges(data = True):
        n1 = e[0]
        n2 = e[1]
        weight = e[2]['weight']

        for i in range(weight):
            letters_list, free_letters = contract_strings(letters_list, n1, n2, free_letters)

    return ','.join(letters_list)

def random_antisym(d, p):
    """
    Inputs: spacetime dimension d and form rank p
    
    Output: a random p-form in d dimensions
    """
    dim_list = [d]*p
    arr = np.random.rand(*dim_list)
    
    G = SymmetricGroup(p)
    out = np.zeros(dim_list)
    
    components = itertools.combinations(range(d), p)
    
    for comp in components:
        this_entry = 0
        
        for g in G.elements:
            this_perm = tuple(g(comp))
            this_entry += arr[this_perm]*g.signature()
        
        for g in G.elements:
            this_perm = tuple(g(comp))
            out[this_perm] = this_entry*g.signature()
            
    return out