import numpy as np
import networkx as nx
from pathlib import Path
from utils import load_pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os


    
def digraph_to_binary_adj(G, dtype=int):
    """
    Convert a networkx DiGraph to a binary adjacency matrix.

    Parameters
    ----------
    G : networkx.DiGraph
        The directed graph to convert. Nodes are expected to be indexed 0..n-1.
    dtype : numpy dtype or int, optional
        dtype for the binary matrix (default int).

    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix where entry (i, j) == 1 if there is an edge i->j.
    """
    nodelist = list(G.nodes())
    num_nodes = len(nodelist)
    adj_bin = np.zeros((num_nodes, num_nodes), dtype=dtype)
    # iterate over edges and inspect optional 'edge_type' attribute
    for u, v, data in G.edges(data=True):
        edge_type = None
        if isinstance(data, dict):
            edge_type = data.get('edge_type', None)
        # default to directed if attribute missing
        if edge_type is None or edge_type == 'directed':
            try:
                adj_bin[int(u), int(v)] = 1
            except Exception:
                # fallback if nodes are not integer indices
                ui = nodelist.index(u)
                vi = nodelist.index(v)
                adj_bin[ui, vi] = 1
        elif str(edge_type) in ('undirected', 'bidirected'):
            try:
                adj_bin[int(u), int(v)] = 1
                adj_bin[int(v), int(u)] = 1
            except Exception:
                ui = nodelist.index(u)
                vi = nodelist.index(v)
                adj_bin[ui, vi] = 1
                adj_bin[vi, ui] = 1
    return adj_bin

def is_SP_and_Tumor_connected(G):
    A = digraph_to_binary_adj(G)

    L = []
    for i in range(A.shape[0]):
        for j in range (A.shape[0]):
            if node_labels[6]=='TMB':
                if (i < 7 and j > 6) and A[i, j] == 1:
                    if [i, j] not in L:#not counting twice undirected/bidirected edge
                        L.append([i, j])
                elif (i > 6 and j <6) and A[j, i] == 1:
                    if [j, i] not in L:
                        L.append([j, i])
            else:
                if (i < 6 and j > 5) and A[i, j] == 1:
                    if [i, j] not in L:#not counting twice undirected/bidirected edge
                        L.append([i, j])
                elif (i > 5 and j < 6) and A[j, i] == 1:
                    if [j, i] not in L:
                        L.append([j, i])
    if len(L)>0:
        return 1, L
    else:
        return 0, L

def edge_type(A, i, j):
    """
    Return edge type for pair (i, j) in matrix A.
    Types :
        - 'none'
        - 'undirected'
        - 'i->j'
        - 'j->i'
    """
    if A[i, j] == 0 and A[j, i] == 0:
        return 'none'
    if A[i, j] == 1 and A[j, i] == 1:
        return 'undirected'
    if A[i, j] == 1 and A[j, i] == 0:
        return 'i->j'
    if A[i, j] == 0 and A[j, i] == 1:
        return 'j->i'
    raise ValueError("Invalid matrix : non-binary values.")


def structural_hamming_distance(A, B):
    """
    Compute Strucutral Hamming distance between two graphs
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must be of the same size.")

    n = A.shape[0]
    shd = 0

    for i in range(n):
        for j in range(i + 1, n):
            type_A = edge_type(A, i, j)
            type_B = edge_type(B, i, j)

            if type_A != type_B:
                shd += 1

    return shd

def analyze_graph_stability_and_direction(graphs_list: list, labels=None):
    """
    Analyzes the structure, stability, and directionality of multiple graphs 
    given as adjacency matrices.

    Args:
        graphs_list (list): A list of networkx DiGraph objects.
        strategy (str): The strategy name for the current set of graphs.
        node_labels (list or dict, optional): Names for the nodes (axis and graph labels).
            If list, should be length n_nodes; if dict, maps node index -> label.
        labels (list, optional): Labels for each graph in the list for reporting.
    """
    
    if labels is None:
        labels = [f"Graph {i+1}" for i in range(len(graphs_list))]

    # Check if the number of matrices matches the number of digraph
    
    # Check if the number of matrices matches the number of labels
    if len(graphs_list) != len(labels):
        print("Error: The number of adjacency matrices does not match the number of labels.")
        return

    print("--- Graph Analysis Summary ---")

    results = []
    for i, G in enumerate(graphs_list):
        label = labels[i]
        A = digraph_to_binary_adj(G)

        is_symmetric = np.allclose(A, A.T)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Compute spectral radius and connectivity
        spectral_radius = None
        is_strongly_connected = None
        try:
            eigenvalues = np.linalg.eigvals(A)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            if not is_symmetric:
                is_strongly_connected = nx.is_strongly_connected(G)
        except np.linalg.LinAlgError:
            spectral_radius = None

        # Skeleton and directionality
        Skeleton_A = np.clip(A + A.T, 0, 1)
        Skeleton_G = G.to_undirected()
        density = nx.density(Skeleton_G)
        total_undirected_edges = Skeleton_G.number_of_edges()
        purely_directed_edges = int(np.abs(A - A.T).sum())//2
        purely_symmetric_edges = total_undirected_edges - purely_directed_edges
        if total_undirected_edges > 0:
            directionality_index = purely_directed_edges / (2 * total_undirected_edges)
        else:
            directionality_index = None
        # 'Smoking' Out-degree
        out_neighbors = np.array(list(G.successors(2)))
        smoking_out_degree_tumor = len(out_neighbors[out_neighbors<7])
        smoking_out_degree_sp = len(out_neighbors[out_neighbors>6])
        #Adjacency between Tumor_Vars and SP
        SP_and_Tumor_connected, adjacencies = is_SP_and_Tumor_connected(G)
        # Print concise summary
        print(f"\n## {label} (Nodes: {n_nodes}, Edges: {n_edges})")
        print(f"* Spectral radius: {spectral_radius}")
        print(f"* Strongly connected: {is_strongly_connected}")
        print(f"* Skeleton density: {density:.4f}")
        print(f"* Directionality index: {directionality_index}")
        print(f"* Smoking Out-degree Tumor: {smoking_out_degree_tumor}")
        print(f"* Smoking Out-degree SP: {smoking_out_degree_sp}")
        print(SP_and_Tumor_connected)
        print(adjacencies)


        results.append({
            'label': label,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'spectral radius': spectral_radius,
            'is_strongly_connected': is_strongly_connected,
            'skeleton density': density,
            'directionality index': directionality_index,
            'smoking out-degree tumor': smoking_out_degree_tumor,
            'smoking out-degree sp': smoking_out_degree_sp,
            'SP and Tumor connected': SP_and_Tumor_connected,
            'SP and Tumor connections': adjacencies,
            'skeleton_matrix': Skeleton_A,
            'adjacency_matrix': A
        })

    return results

def visualize_aggregate_adjacencies(adj_matrices_list: list, strategy: str, node_labels: list = None, pos: dict = None,
                                   save_plots: bool = True, out_dir: str = None, fmt: str = 'png', dpi: int = 150,
                                   log_scale: bool = False, label_show_threshold: int = 30, show_edge_labels = False, show_plots: bool = True):
    """
    Visualizes the aggregate connectivity across a population of adjacency matrices,
    both in an undirected and directed manner.

    Args:
        adj_matrices_list (list): A list of NumPy adjacency matrices (np.array).
        strategy (str): The strategy name for the current set of graphs.
        node_labels (list or dict, optional): Names for the nodes (axis and graph labels).
            If list, should be length n_nodes; if dict, maps node index -> label.
        pos (dict, optional): Precomputed positions for nodes in the graph visualizations.
    """
    
    if not adj_matrices_list:
        print("Error: The adjacency matrix list is empty.")
        return

    num_graphs = len(adj_matrices_list)
    num_nodes = adj_matrices_list[0].shape[0]

    # Initialize aggregate matrices
    # Sum of original (directed) versions for directed view
    aggregate_directed_matrix = np.zeros((num_nodes, num_nodes))

    for i, A in enumerate(adj_matrices_list):
        if A.shape != (num_nodes, num_nodes):
            print(f"Warning: Graph {i+1} has a different number of nodes ({A.shape[0]}) "
                  f"than the first graph ({num_nodes}). Skipping this graph for aggregation.")
            continue
        
        # For directed view: Add the original (potentially directed) matrix
        aggregate_directed_matrix += A

    # Symmetric versions for undirected view
    aggregate_undirected_matrix = aggregate_directed_matrix + aggregate_directed_matrix.T

    print("\n--- Aggregate Adjacency Visualization ---")
    # Aesthetics
    sns.set_style('whitegrid')
    palette_nodes = sns.color_palette('Set2')

    # Prepare output directory
    if save_plots:
        if out_dir is None:
            out_dir = Path(__file__).resolve().parent / 'plots'
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Undirected Edge Frequency Map ---
    # adaptive fonts and label visibility
    if num_nodes <= 10:
        title_fontsize = 14
        label_fontsize = 10
        ann_font = 9
    elif num_nodes <= 20:
        title_fontsize = 13
        label_fontsize = 9
        ann_font = 8
    elif num_nodes <= 40:
        title_fontsize = 12
        label_fontsize = 8
        ann_font = 7
    else:
        title_fontsize = 11
        label_fontsize = 6
        ann_font = 6

    show_labels = num_nodes <= label_show_threshold

    plt.figure(figsize=(8, 7))
    im = plt.imshow(np.log1p(aggregate_undirected_matrix) if log_scale else aggregate_undirected_matrix,
                    cmap='viridis', origin='upper')
    cbar = plt.colorbar(im)
    im = plt.imshow(np.log1p(aggregate_undirected_matrix) if log_scale else aggregate_undirected_matrix,
                    cmap='viridis', origin='upper')
    plt.title(f'Undirected Edge Frequency, Number of graphs: {num_graphs}, Strategy: {strategy}', fontsize=title_fontsize)
    plt.xlabel('Node', fontsize=label_fontsize)
    plt.ylabel('Node', fontsize=label_fontsize)

    # Prepare tick labels if provided (supports list or dict)
    if node_labels is None:
        xlabels = ylabels = [str(i) for i in range(num_nodes)]
    elif isinstance(node_labels, dict):
        # dict mapping index -> label
        xlabels = ylabels = [node_labels.get(i, str(i)) for i in range(num_nodes)]
    else:
        # assume list-like
        xlabels = ylabels = [str(x) for x in node_labels]

    if show_labels:
        plt.xticks(np.arange(num_nodes), xlabels, rotation=90, fontsize=label_fontsize)
        plt.yticks(np.arange(num_nodes), ylabels, fontsize=label_fontsize)
    else:
        plt.xticks([])
        plt.yticks([])

    # Annotate numeric values on heatmap
    max_val = np.max(aggregate_undirected_matrix) if aggregate_undirected_matrix.size else 0
    thresh = max_val / 2.0 if max_val > 0 else 0
    for (i, j), val in np.ndenumerate(aggregate_undirected_matrix):
        if val != 0:
            color = 'white' if val > thresh else 'black'
            if ann_font >= 6:
                plt.text(j, i, f"{int(val)}", ha='center', va='center', color=color, fontsize=ann_font)

    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"{strategy}_undirected_heatmap.png"
        fname_pdf = out_path / f"{strategy}_undirected_heatmap.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    if show_plots:
        plt.show()
    # Trigger image for Undirected Edge Frequency Map
    

    # --- 2. Directed Edge Frequency Map ---
    plt.figure(figsize=(8, 7))
    im2 = plt.imshow(np.log1p(aggregate_directed_matrix) if log_scale else aggregate_directed_matrix, cmap='plasma', origin='upper')
    cbar2 = plt.colorbar(im2)
    cbar2.set_label('Number of Graphs with Directed Edge (Row -> Column)' + (' (log1p)' if log_scale else ''))
    plt.title(f'Directed Edge Frequency, Number of graphs: {num_graphs}, Strategy: {strategy}', fontsize=title_fontsize)
    plt.xlabel('Target Node (j)', fontsize=label_fontsize)
    plt.ylabel('Source Node (i)', fontsize=label_fontsize)

    plt.xticks(np.arange(num_nodes), xlabels, rotation=90)
    plt.yticks(np.arange(num_nodes), ylabels)

    # Annotate numeric values on directed heatmap
    max_val2 = np.max(aggregate_directed_matrix) if aggregate_directed_matrix.size else 0
    thresh2 = max_val2 / 2.0 if max_val2 > 0 else 0
    for (i, j), val in np.ndenumerate(aggregate_directed_matrix):
        if val != 0:
            color = 'white' if val > thresh2 else 'black'
            plt.text(j, i, f"{int(val)}", ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"{strategy}_directed_heatmap.png"
        fname_pdf = out_path / f"{strategy}_directed_heatmap.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    if show_plots:
        plt.show()
    # Trigger image for Directed Edge Frequency Map

    # --- Graph visualizations at thresholds (skeleton and directed) ---
    # Compute a consistent layout using the aggregate undirected matrix as base
    if pos is None:
        try:
            base_graph = nx.from_numpy_array(aggregate_undirected_matrix)
            pos = nx.spring_layout(base_graph, seed=42)
        except Exception:
            # fallback to default positions
            pos = {i: (np.cos(2.0 * np.pi * i / num_nodes), np.sin(2.0 * np.pi * i / num_nodes)) for i in range(num_nodes)}

    thresholds = [0.5, 0.7]
    if num_nodes > 7:
        fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for idx, thresh in enumerate(thresholds):
        # Directed edges at threshold
        A_dir = ((aggregate_directed_matrix / float(num_graphs)) >= thresh).astype(int)

        # Skeleton (undirected) at threshold
        A_sk = np.clip(A_dir + A_dir.T, 0, 1)
        G_sk = nx.Graph(A_sk)

        ax = axes[idx]
        ax.set_title(f'Skeleton ({int(thresh*100)}% consensus)', fontsize=20)
        
        # Color the nodes: firsts 7 different if >= 7 nodes in total
        if num_nodes >= 7:
            node_colors = ["#6BE6FF" if i < 7 else palette_nodes[0] for i in range(num_nodes)]
        else:
            node_colors = palette_nodes[0]
        nx.draw_networkx_nodes(G_sk, pos, ax=ax, node_size=800, node_color=node_colors)
        
        if show_labels:
            nx.draw_networkx_labels(G_sk, pos, labels={i: xlabels[i] for i in range(num_nodes)}, font_size=max(10, label_fontsize+2), ax=ax)
        nx.draw_networkx_edges(G_sk, pos, ax=ax, edge_color='gray', width=2, connectionstyle='arc3,rad=0.2', node_size=800)
        ax.axis('off')

        G_dir = nx.DiGraph(A_dir)
        ax = axes[idx + 2]
        ax.set_title(f'Directed edges ({int(thresh*100)}% consensus)', fontsize=20)
        
        # Color the nodes: firsts 7 different if >= 7 nodes in total
        if num_nodes >= 7:
            if num_nodes==19:
                node_colors = ["#FFAB6B" if i < 7 else palette_nodes[1] for i in range(num_nodes)]
            else:
                node_colors = ["#FFAB6B" if i < 6 else palette_nodes[1] for i in range(num_nodes)]

        else:
            node_colors = palette_nodes[1]
        nx.draw_networkx_nodes(G_dir, pos, ax=ax, node_size=800, node_color=node_colors)
        
        if show_labels:
            nx.draw_networkx_labels(G_dir, pos, labels={i: xlabels[i] for i in range(num_nodes)}, font_size=max(10, label_fontsize+2), ax=ax)
        # draw directed edges with arrows
        nx.draw_networkx_edges(G_dir, pos, ax=ax, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20, width=2, connectionstyle='arc3,rad=0.2', node_size=800)
        ax.axis('off')

    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"{strategy}_graphs_thresholds.png"
        fname_pdf = out_path / f"{strategy}_graphs_thresholds.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    if show_plots:
        plt.show()

    #plot of consensus causal model by strategy
    consensus = aggregate_directed_matrix/float(num_graphs)
    consensus_u = np.zeros_like(consensus)
    for i in range(consensus.shape[0]):
        for j in range(consensus.shape[0]):
            if consensus[i, j]>consensus[j ,i]:
                consensus[j, i] = 0
            if consensus[i ,j] == consensus[j, i]:
                consensus_u[i,j] = consensus[i,j]
                consensus_u[j,i] = consensus[j,i]
    G_cons = nx.DiGraph(consensus)
    G_cons_u = nx.Graph(consensus_u)
    for i in range(num_nodes):
        for j in range(num_nodes):
            w=consensus[i,j]
            v=consensus[j,i]
            if w == v:
                if G_cons.has_edge(i,j):
                    G_cons.remove_edge(i, j)
                if G_cons.has_edge(j, i):
                    G_cons.remove_edge(j, i)
            else:
                if G_cons_u.has_edge(i, j):
                    G_cons_u.remove_edge(i, j)
                if G_cons_u.has_edge(j, i): 
                    G_cons_u.remove_edge(j, i)
    print('G_cons_u', G_cons_u) 
    # Get weights for coloring
    weights = nx.get_edge_attributes(G_cons, 'weight')
    edge_labels = {edge: f"{w:.1f}" for edge, w in weights.items()}
    w_values = np.array(list(weights.values()))

    if len(w_values) > 0:
        w_min, w_max = w_values.min(), w_values.max()
        norm = (w_values - w_min) / (w_max - w_min + 1e-9)
        colors = [(1 - v, 1 - v, 1 - v) for v in norm]  
    else:
        colors = []

    # Graph plot
    if num_nodes >7:
        plt.figure(figsize=(15, 15))
    else:
        plt.figure(figsize=(9,9))
    nx.draw_networkx_nodes(G_cons, pos, node_size=1000, node_color=node_colors)
    nx.draw_networkx_labels(G_cons, pos, labels={i: node_labels[i] for i in range(num_nodes)})

    # Edges plot with color
    #Directed
    nx.draw_networkx_edges(G_cons, pos, edge_color=colors, width=2, arrowsize=20)
    #Undirected
    nx.draw_networkx_edges(G_cons_u, pos, edge_color=colors, width=2, arrowsize=20)

    # Plot weights
    if show_edge_labels:
        nx.draw_networkx_edge_labels(G_cons, pos, edge_labels=edge_labels)

    plt.axis("off")
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"gradient_{strategy}_graphs_thresholds.png"
        fname_pdf = out_path / f"gradient_{strategy}_graphs_thresholds.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    if show_plots:
        plt.show()
    
    
    if num_nodes <8:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # -------Comparison single graph vs consensus (PC chisq)----------
    #single causal graph construction
        Gd = nx.DiGraph()   # directed
        Gu = nx.Graph()     # undirected

        Gd.add_edge(0, 5)
        Gu.add_edge(0, 3)

        Gd.add_edge(1, 2)
        Gu.add_edge(1, 4)

        Gd.add_edge(3, 2)

        Gd.add_edge(4, 5)

        #single graph plot#
        nx.draw_networkx_nodes(Gd, pos, node_color=node_colors[:-1], node_size=1000, ax=axes[0])
        nx.draw_networkx_labels(Gd, pos, ax=axes[0], labels={i: node_labels[i] for i in range(num_nodes-1)})

        # directed edges
        nx.draw_networkx_edges(Gu, pos, ax=axes[0], width=2, edge_color = 'grey', style= 'dashed', min_source_margin=15,
        min_target_margin=15)
        # undirected edges
        nx.draw_networkx_edges(Gd, pos, ax=axes[0], width=2, edge_color="grey", arrows=True, arrowsize=20, min_source_margin=15,
        min_target_margin=15)

        axes[0].axis("off")
        #consensus plot
        nx.draw_networkx_nodes(G_cons, pos, ax=axes[1], node_size=1000, node_color=node_colors)
        nx.draw_networkx_labels(G_cons, pos, ax=axes[1], labels={i: node_labels[i] for i in range(num_nodes)})

        nx.draw_networkx_edges(G_cons, pos, ax=axes[1], edge_color=colors, width=2, arrowsize=20, min_source_margin=15,
        min_target_margin=15)
        nx.draw_networkx_edges(G_cons_u, pos, ax=axes[1], edge_color=colors, width=2, arrowsize=20, style= 'dashed', min_source_margin=15,
        min_target_margin=15)

        axes[1].axis("off")
    else:#Full data
        fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    # -------Comparison single graph vs consensus (PC chisq)----------
        Gd = nx.DiGraph()   
        Gu = nx.Graph()  

        Gd.add_edge(0, 5)
        Gu.add_edge(0, 3)

        Gd.add_edge(1, 2)

        Gd.add_edge(3, 2)

        Gd.add_edge(4, 5)

        Gd.add_edge(2, 15)

        Gd.add_edge(7, 15)
        Gd.add_edge(8,12)
        Gu.add_edge(9,17)
        Gd.add_edge(10, 13)
        Gd.add_edge(11,15)
        Gd.add_edge(11,13)
        Gd.add_edge(12, 13)
        Gd.add_edge(13,15)
        Gd.add_edge(14, 12)
        
        #single graph plot#

        nx.draw_networkx_nodes(G_cons, pos, node_color=node_colors, node_size=1000, ax=axes[0])
        nx.draw_networkx_labels(G_cons, pos, ax=axes[0], labels={n: node_labels[n] for n in range(num_nodes)})

        
        nx.draw_networkx_edges(Gu, pos, ax=axes[0], width=2, edge_color = 'grey', style= 'dashed', min_source_margin=15,
        min_target_margin=15)
        
        nx.draw_networkx_edges(Gd, pos, ax=axes[0], width=2, edge_color="grey", arrows=True, arrowsize=20, min_source_margin=15,
        min_target_margin=15)

        axes[0].axis("off")
        #consensus plot
        nx.draw_networkx_nodes(G_cons, pos, ax=axes[1], node_size=1000, node_color=node_colors)
        nx.draw_networkx_labels(G_cons, pos, ax=axes[1], labels={i: node_labels[i] for i in range(num_nodes)})

        nx.draw_networkx_edges(G_cons, pos, ax=axes[1], edge_color=colors, width=2, arrowsize=20, min_source_margin=15,
        min_target_margin=15)
        nx.draw_networkx_edges(G_cons_u, pos, ax=axes[1], edge_color=colors, width=2, arrowsize=20, style= 'dashed', min_source_margin=15,
        min_target_margin=15)

        axes[1].axis("off")
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"PC_fisherz_vs_consensus_{strategy}_graphs.png" #old name (not PC fisherz anymore)
        fname_pdf = out_path / f"PC_fisherz__vs_consensus_{strategy}_graphs.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved {fname_png} and {fname_pdf}")
    if show_plots:
        plt.show()
        
    return pos


def aggregate_and_visualize_results(Results: dict, node_labels=None, figsize=(14, 6), save_plots: bool = True, 
                                    out_dir: str = None, fmt: str = 'png', dpi: int = 150, show_plots: bool = True):
    """
    Compute aggregated statistics from the Results dictionary and plot
    a set of summary visualizations.

    Parameters
    ---------
    Results : dict
        Mapping strategy -> list of per-graph result dicts produced by
        `analyze_graph_stability_and_direction`.
    node_labels : list or dict, optional
        Labels for nodes used in adjacency heatmaps.
    figsize : tuple
        Default figure size for plots.

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame with one row per graph and columns with computed metrics.
    """
    # Flatten Results into a DataFrame
    rows = []
    for strategy, res_list in Results.items():
        for r in res_list:
            edges = r.get('SP and Tumor connections')
            if not isinstance(edges, list):
                edges = []
            rows.append({
                'strategy': strategy,
                'label': r.get('label'),
                'n_nodes': r.get('n_nodes'),
                'n_edges': r.get('n_edges'),
                'spectral radius': r.get('spectral radius'),
                'skeleton density': r.get('skeleton density'),
                'directionality index': r.get('directionality index'),
                'is_strongly_connected': bool(r.get('is_strongly_connected')) if r.get('is_strongly_connected') is not None else False,
                'smoking out-degree tumor': r.get('smoking out-degree tumor'),
                'smoking out-degree sp': r.get('smoking out-degree sp'),
                'pcd_betweenness': r.get('pcd_betweenness'),
                'SP and Tumor connected': r.get('SP and Tumor connected'),
                'SP and Tumor connections': edges,
                'adjacency_matrix': r.get('adjacency_matrix')
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print('No results to aggregate.')
        return df
    # Convert types and handle NaNs
    numeric_cols = ['n_nodes', 'n_edges', 'spectral radius', 'skeleton density', 'directionality index', 'smoking out degree', 'pcd_betweenness', 'SP and Tumor connected']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Prepare output dir when saving
    if save_plots:
        if out_dir is None:
            out_dir = Path(__file__).resolve().parent / 'plots'
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    # 1) Boxplots for spectral radius, skeleton density, directionality index by strategy
    metrics = ['spectral radius', 'skeleton density', 'directionality index']
    plt.figure(figsize=(figsize[0], 5))
    strategy_order = df['strategy'].unique().tolist()
    for i, m in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='strategy', y=m, data=df, palette='Set2', order=strategy_order)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(m, fontsize=20)
        plt.xlabel('')  
        plt.ylabel('') 
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"metrics_boxplots.png"
        fname_pdf = out_path / f"metrics_boxplots.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved metrics boxplots to {out_path}")
    if show_plots:
        plt.show()

    # 2) Fraction of strongly connected graphs per strategy
    conn = df.groupby('strategy')['is_strongly_connected'].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x='strategy', y='is_strongly_connected', data=conn, palette='Set2', order=strategy_order)
    # plt.ylim(0, 1)
    plt.xlabel('')  
    plt.ylabel('')  
    plt.xticks(rotation=45, fontsize=12, ticks=range(len(strategy_order)),
    labels=[f"St.{i+1}" for i in range(len(strategy_order))])
    plt.yticks(fontsize=10)
    # plt.title('Connectivity fraction by strategy')
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"connectivity_fraction.png"
        fname_pdf = out_path / f"connectivity_fraction.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved connectivity fraction plot to {out_path}")
    if show_plots:
        plt.show()

    # 3) Heatmaps: mean adjacency per strategy
    strategies = list(Results.keys())
    mean_adjs = {}
    for strat in strategies:
        mats = [r['adjacency_matrix'] for r in Results[strat] if 'adjacency_matrix' in r]
        if not mats:
            continue
        # compute mean adjacency (counts) across matrices
        mean_adj = np.mean(np.stack(mats, axis=0), axis=0)
        
        mean_adj = np.round(mean_adj, 1)
        mean_adjs[strat] = mean_adj

    if mean_adjs:
        import matplotlib.gridspec as gridspec
        n_strat = len(mean_adjs)
        n_cols = min(n_strat, 2)
        n_rows = int(np.ceil(n_strat / n_cols))
        fig = plt.figure(figsize=(n_cols * 10 + 1, n_rows * 10))
        gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols+1, width_ratios=[1]*n_cols + [0.05], wspace=0.3)
        titles = ['St.1', 'St.2', 'St.3', 'St.4']
        axes = []
        ims = []
        for idx, (strat, mat) in enumerate(mean_adjs.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.grid(False) 
            axes.append(ax)
            num_nodes = mat.shape[0]
            # adaptive fonts and label visibility
            if num_nodes <= 10:
                label_fontsize = 25
                ann_font = 20
            elif 10 < num_nodes <= 20:
                label_fontsize = 25
                ann_font = 3
            elif 20 < num_nodes <= 40:
                label_fontsize = 8
                ann_font = 3
            else:
                label_fontsize = 6
                ann_font = 3

            show_labels = num_nodes <= 30

            im = ax.imshow(mat, cmap='viridis', origin='upper')
            ims.append(im)
            ax.set_title(titles[idx], fontsize=max(30, label_fontsize+2))
            
            # tick labels
            if show_labels:
                if node_labels is None:
                    ticks = [str(i) for i in range(num_nodes)]
                elif isinstance(node_labels, dict):
                    ticks = [node_labels.get(i, str(i)) for i in range(num_nodes)]
                else:
                    ticks = [str(x) for x in node_labels]
                ax.set_xticks(np.arange(num_nodes))
                ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=label_fontsize)
                ax.set_yticks(np.arange(num_nodes))
                ax.set_yticklabels(ticks, fontsize=label_fontsize)
                
                # display y ticks only for first column
                if col != 0:
                    ax.set_yticklabels([])
                
                
                # if row != n_rows - 1:
                #     ax.set_xticklabels([])
                ax.set_xticklabels([])
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            # annotate a subset if small
            if ann_font >= 6:
                for (ii, jj), val in np.ndenumerate(mat):
                    if val != 0:
                        ax.text(jj, ii, f"{val:.1f}", ha='center', va='center', color='white' if val > 0.5 else 'black', fontsize=ann_font)
        
        cax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(ims[0], cax=cax)
        cbar.ax.tick_params(labelsize=20)
        plt.tight_layout(rect=[0, 0, 0.97, 1])
        if save_plots:
            fname_png = out_path / f"mean_adj.png"
            fname_pdf = out_path / f"mean_adj.pdf"
            plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
            plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
            print(f"Saved mean adjacency heatmaps to {out_path}")
        if show_plots:
            plt.show()

    # 4) Scatter: skeleton density vs directionality index
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x='skeleton density', y='directionality index', hue='strategy', palette='tab10')
    plt.title('Skeleton density vs Directionality Index')
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"skeleton_density_vs_directionality.png"
        fname_pdf = out_path / f"skeleton_density_vs_directionality.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved skeleton density vs directionality plot to {out_path}")
    if show_plots:
        plt.show()

    # 5) average 'Smoking' Out-degree
    AGG = "mean"      # ou "mean"

    agg_df = df.groupby("strategy").agg({
        "smoking out-degree tumor": AGG,
        "smoking out-degree sp": AGG
    }).reset_index()

    agg_long = agg_df.melt(
        id_vars="strategy",
        value_vars=["smoking out-degree tumor", "smoking out-degree sp"],
        var_name="type",
        value_name="outdegree"
    )  

    agg_long["type"] = agg_long["type"].replace({
        "smoking out-degree tumor": "Patient and Tumor",
        "smoking out-degree sp": "Cellular Pathways"
    })

    plt.figure(figsize=(7, 5))
    sns.barplot(
    data=agg_long,
        x="strategy",
        y="outdegree",
        hue="type",
        palette="Set2",
        order=strategy_order  
    )

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, fontsize=15, ticks=range(len(strategy_order)),
    labels=[f"St.{i+1}" for i in range(len(strategy_order))])
    plt.yticks(fontsize=10)
    plt.legend(title='', fontsize=13)

    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"Avg Smoking Out-degree.png"
        fname_pdf = out_path / f"Avg Smoking Out-degree.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved Avg Smoking Out-degree plot to {out_path}")
    if show_plots:
        plt.show()

    # 6) Max 'Smoking' Out-degree
    AGG = "max"      

    agg_df = df.groupby("strategy").agg({
        "smoking out-degree tumor": AGG,
        "smoking out-degree sp": AGG
    }).reset_index()

    agg_long = agg_df.melt(
        id_vars="strategy",
        value_vars=["smoking out-degree tumor", "smoking out-degree sp"],
        var_name="type",
        value_name="outdegree"
    )  

    agg_long["type"] = agg_long["type"].replace({
        "smoking out-degree tumor": "Tumoral",
        "smoking out-degree sp": "SP"
    })

    plt.figure(figsize=(7, 5))
    sns.barplot(
    data=agg_long,
        x="strategy",
        y="outdegree",
        hue="type",
        palette="Set2",
        order=strategy_order 
    )

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='', fontsize=10)

    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"Max Smoking Out-degree.png"
        fname_pdf = out_path / f"Max Smoking Out-degree.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved Max Smoking Out-degree plot to {out_path}")
    if show_plots:
        plt.show()

    # 7) Average 'Programmed Cell Death' Betweenness
    if num_nodes>7:
        avg_btw = df.groupby('strategy')['pcd_betweenness'].mean().reset_index()
        plt.figure(figsize=(6, 5))
        sns.barplot(
        x='strategy',
        y='pcd_betweenness',
        data=avg_btw,
        palette='Set2',
        order=strategy_order
        )
        plt.xlabel('')  
        plt.ylabel('') 
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_plots:
            fname_png = out_path / f"Average betweenness.png"
            fname_pdf = out_path / f"AVerage betweenness.pdf"
            plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
            plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
            print(f"Saved Average betweenness plot to {out_path}")
        if show_plots:
            plt.show()

        #8) Max 'Programmed Cell Death' Betweenness
        max_btw = df.groupby('strategy')['pcd_betweenness'].max().reset_index()
        plt.figure(figsize=(6, 5))
        sns.barplot(
        x='strategy',
        y='pcd_betweenness',
        data=max_btw,
        palette='Set2',
        order=strategy_order
        )
        plt.xlabel('') 
        plt.ylabel('')
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_plots:
            fname_png = out_path / f"Max betweenness.png"
            fname_pdf = out_path / f"Max betweenness.pdf"
            plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
            plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
            print(f"Saved Max betweenness plot to {out_path}")
        if show_plots:
            plt.show()
    #9) Fractions of graphs with SP and Tumors_vars connected
        fraction_of_connected = df.groupby('strategy')['SP and Tumor connected'].mean().reset_index()
        print(fraction_of_connected)
        plt.figure(figsize=(6, 5))
        sns.barplot(
            x='strategy',
            y='SP and Tumor connected',
            data=fraction_of_connected,
            palette='Set2',
            order=strategy_order
            )
        plt.xlabel('')  
        plt.ylabel('') 
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        if save_plots:
            fname_png = out_path / f"SP and tumor connected.png"
            fname_pdf = out_path / f"SP and tumor connected.pdf"
            plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
            plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
            print(f"Saved SP and tumor connected plot to {out_path}")
        if show_plots:
            plt.show()
        #10) most frequent connection between SP and Tumor variables    
        from collections import Counter
        import math

        strategies = sorted(df["strategy"].unique())

        # determine grid shape
        n = len(strategies)
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
        axes = axes.flatten()

        for idx, strategy in enumerate(strategies):
            ax = axes[idx]

            subdf = df[df["strategy"] == strategy]
            n_graphs = len(subdf)    
            all_edges = []

            # Collect all edges for this strategy
            for edges in subdf['SP and Tumor connections']:
                if edges is None or isinstance(edges, float):
                    continue

                if not isinstance(edges, (list, tuple)):
                    continue

                for e in edges:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        all_edges.append(tuple(e))
            counter = Counter(all_edges)
            most_common = counter.most_common(5)

            if not most_common:
                ax.set_title(f"{strategy} (aucune arête)")
                ax.axis("off")
                continue

            # --- label conversion node_name[i] -> node_name[j] ---
            labels = [f"{node_labels[i]} → {node_labels[j]}" for (i, j), _ in most_common]

            # --- value = perrcentage ---
            values = [(count / n_graphs) * 100 for (_, count) in most_common]

            ax.bar(labels, values)
            #ax.set_title(f"{strategy, n_graphs}")

            ax.set_ylabel("")
            ax.tick_params(axis='x', rotation=45, labelsize = 14)

        # hide unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle("Most common edges connecting SP and Tumor variables (%)", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # ----- Save-----
        if save_plots:
            fname_png = out_path / f"SP_and_tumor_connections_subplots.png"
            fname_pdf = out_path / f"SP_and_tumor_connections_subplots.pdf"
            fig.savefig(fname_png, dpi=dpi, bbox_inches='tight')
            fig.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')

        # ----- Plot -----
        if show_plots:
            plt.show()

    #11) SHD by strategy
    def compute_mean_shd_per_graph(df):
        """"    
        Adds a 'shd' column to the DataFrame, containing for each graph
        the mean SHD compared to the other graphs of the same strategy.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Must contain 'strategy' and 'adjacency_matrix'.
        shd_func : callable
            SHD function SHD(A, B) -> float.

        Returns
        -------
        df : DataFrame with an additional 'shd' column.
        """
        df = df.copy()
        shd_values = []

        # strategy loop
        for strategy, group in df.groupby('strategy'):
            mats = group['adjacency_matrix'].tolist()
            n = len(mats)

            # for each graph by group
            for i in range(n):
                # all SHD
                dists = [
                    structural_hamming_distance(mats[i], mats[j])
                    for j in range(n) if j != i
                ]
                shd_values.append(np.mean(dists))

        df['shd'] = shd_values
        return df

    df = compute_mean_shd_per_graph(df)
    metrics = ['shd', 'skeleton density', 'directionality index',  'spectral radius']
    titles = ['SHD', 'Skeleton Density', 'Directionality Index', 'Spectral Radius']
    plt.figure(figsize=(figsize[0], 10))
    strategy_order = df['strategy'].unique().tolist()
    for i, m in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x='strategy', y=m, data=df, palette='Set2', order=strategy_order)
        sns.stripplot(x='strategy', y=m, data=df, palette='Set2', order=strategy_order, alpha=0.7,
        size=5,
        edgecolor='black',
        linewidth=0.5)
        plt.xticks(rotation=45, fontsize=20, ticks=range(len(strategy_order)),
        labels=[f"St.{i+1}" for i in range(len(strategy_order))])
        plt.yticks(fontsize=15)
        plt.title(titles[i-1], fontsize=22)
        plt.xlabel('')  
        plt.ylabel('') 
    plt.tight_layout()
    if save_plots:
        fname_png = out_path / f"metrics_boxplots.png"
        fname_pdf = out_path / f"metrics_boxplots.pdf"
        plt.savefig(fname_png, dpi=dpi, bbox_inches='tight')
        plt.savefig(fname_pdf, dpi=dpi, bbox_inches='tight')
        print(f"Saved metrics boxplots to {out_path}")
    if show_plots:
        plt.show()
    plt.figure(figsize=(figsize[0], 5))

    return df

def deduplicate_graphs(graphs_dict):
    """
    Deduplicate graphs based on their adjacency matrices, treating
    methods run with priors separately from those without.
    Groups by (method base, prior_status) and keeps only unique
    adjacency matrices within each group, removing redundant results
    from the same method with different specifications.
    """
    from collections import defaultdict
    
    dedup_dict = {}
    # Key: (base_method, prior_status), Value: list of (original_key, graph)
    method_prior_groups = defaultdict(list)
    
    # 1. Group results by base method AND prior status
    for key, graph in graphs_dict.items():
        parts = key.split('_')
        base_method = parts[0]
        
        # Check for prior status
        prior_status = 'prior' if key.endswith('_prior') else 'no_prior'
        
        # Create a unique group key combining base method and prior status
        group_key = (base_method, prior_status)
        method_prior_groups[group_key].append((key, graph))
        # NOTE: if methods run with prior have to be considered in the same pool 
        # as those run without prior, group_key=base_method
    
    # 2. For each group, deduplicate by adjacency matrix
    for group_key, group in method_prior_groups.items():
        # unique_matrices is keyed by the hashable adjacency matrix tuple
        # Value is (original_key, graph)
        unique_matrices = {}
        
        for key, graph in group:
            adj_mat = digraph_to_binary_adj(graph)
            adj_mat_tuple = tuple(adj_mat.flatten())  # Convert to hashable tuple

            # Keep the first occurrence of a unique matrix
            if adj_mat_tuple not in unique_matrices:
                unique_matrices[adj_mat_tuple] = (key, graph)
        
        # 3. Add deduplicated graphs to result
        for (key, graph) in unique_matrices.values():
            dedup_dict[key] = graph
            
    return dedup_dict

if __name__ == '__main__':
    # parse CLI
    parser = argparse.ArgumentParser(description='Run causallearn benchmarks on selected dataset')
    parser.add_argument('--dataset', choices=['tumor', 'full', 'full_wo_tmb'], default='tumor',
                        help="Choose 'tumor' to analyse results on tumor datasets or 'full' for the full dataset collection")
    args = parser.parse_args()

    print(f"Analysing results for dataset: {args.dataset}")

    # Select lists depending on requested dataset
    node_labels={0: 'Age', 1: 'Sex', 2: 'Smoking', 3: 'SPY', 4: 'Stage', 5: 'Status', 6: 'TMB'}
    if args.dataset == 'tumor':
        key_list = ['Tumor_data_mixed', 'Tumor_data_mixed_no_nan', 
                    'Tumor_data_mixed_imputed', 'Tumor_data_discrete_imputed']
        results_dir = Path(__file__).resolve().parent/'results/tumor_data'
        out_dir=Path(__file__).resolve().parent/'plots/tumor_data'
    elif args.dataset == 'full':  # full dataset collection
        key_list = ['Full_continuous', 'Full_continuous_no_nan', 
                    'Full_continuous_imputed', 'Full_discrete_imputed']
        # pthws_labels = {7: 'Cell-Cell', 8: 'Cellular res',
        #                9: 'Chromatin org.', 10: 'Developm. Bio.',
        #                11: 'Extracellular org.', 12: 'Gene expression',
        #                13: 'Immune System', 14: 'Metabolism prot.',
        #                15: 'Neuronal System', 16: 'Programmed Cell Death',
        #                17: 'Signal Transduction', 18: 'Vesicle-med. transport'
        #                }
        pthws_labels = {7: 'SP1', 8: 'SP2',
                       9: 'SP3', 10: 'SP4',
                       11: 'SP5', 12: 'SP6',
                       13: 'SP7', 14: 'SP8',
                       15: 'SP9', 16: 'SP10',
                       17: 'SP11', 18: 'SP12'
                       }
        
        node_labels.update(pthws_labels)
        results_dir = Path(__file__).resolve().parent/'results/full_data'
        out_dir=Path(__file__).resolve().parent/'plots/full_data'
    elif args.dataset == 'full_wo_tmb':  # full dataset collection
        key_list = ['Full_continuous_wo_tmb', 'Full_continuous_wo_tmb_no_nan', 
                    'Full_continuous_wo_tmb_imputed']
        # pthws_labels = {7: 'Cell-Cell', 8: 'Cellular res',
        #                9: 'Chromatin org.', 10: 'Developm. Bio.',
        #                11: 'Extracellular org.', 12: 'Gene expression',
        #                13: 'Immune System', 14: 'Metabolism prot.',
        #                15: 'Neuronal System', 16: 'Programmed Cell Death',
        #                17: 'Signal Transduction', 18: 'Vesicle-med. transport'
        #                }
        pthws_labels = {6: 'SP1', 7: 'SP2',
                       8: 'SP3', 9: 'SP4',
                       10: 'SP5', 11: 'SP6',
                       12: 'SP7', 13: 'SP8',
                       14: 'SP9', 15: 'SP10',
                       16: 'SP11', 17: 'SP12'
                       }
        node_labels.update(pthws_labels)
        results_dir = Path(__file__).resolve().parent/'results/full_data_wo_tmb'
        out_dir=Path(__file__).resolve().parent/'plots/full_data_wo_tmb'
    
    out_dir.mkdir(parents=True, exist_ok=True)

    Outputs_tumor_data = {}
    for key in key_list:
        Output = load_pickle(results_dir/f"Outputs_{key}.pkl")
        Outputs_tumor_data.update(Output)
    
    with os.scandir(out_dir) as it:
        if any(it):
            print(f"Results for dataset: {args.dataset} already exist in {out_dir}. \
                  Please remove or move the existing plots directory before running the analysis.")
            print(Outputs_tumor_data)
            exit(1)
    
    Mix_strategy=['All','All_wo_Prior','All_Prior','Original','Removed_NaN','Imputed_all','Imputed_mixed','Imputed_discrete']
    Datasets = ['Data_with_missing','Data_no_nan','Data_imputed']
    Methods = ['PC','FCI','GES','ExactBIC','DirectLiNGAM']

    all,all_prior,all_wo_prior,original,removed_nan,imputed_all,imputed_mixed,imputed_discrete = [],[],[],[],[],[],[],[]
    all_m,all_prior_m,all_wo_prior_m,original_m,removed_nan_m,imputed_all_m,imputed_mixed_m,imputed_discrete_m = [],[],[],[],[],[],[],[]
    if args.dataset == 'full_wo_tmb':
        del Outputs_tumor_data['Data_imputed']['ExactBIC_dp']
    for Dataset in Datasets:
        Out = Outputs_tumor_data[Dataset]
        # Deduplicate graphs within each dataset
        Out_dedup = deduplicate_graphs(Out)
        for key,value in Out_dedup.items():
            adj_mat = digraph_to_binary_adj(value)
            all_m.append(adj_mat)
            all.append(value)
            if 'prior' in key:
                all_prior_m.append(adj_mat)
                all_prior.append(value)
            else:
                all_wo_prior_m.append(adj_mat)
                all_wo_prior.append(value)
            if Dataset == 'Data_with_missing':
                original_m.append(adj_mat)
                original.append(value)
            if Dataset == 'Data_no_nan':
                removed_nan_m.append(adj_mat)
                removed_nan.append(value)
            if Dataset == 'Data_imputed':
                imputed_all_m.append(adj_mat)
                imputed_mixed_m.append(adj_mat)
                imputed_all.append(value)
                imputed_mixed.append(value)
            if Dataset == 'Data_discrete':
                imputed_all_m.append(adj_mat)
                imputed_discrete_m.append(adj_mat)
                imputed_all.append(value)
                imputed_discrete.append(value)
    Dict_adj_matrices = {
        'All': {'matrices':all_m, 'graphs': all},
        'All with prior': {'matrices':all_prior_m, 'graphs': all_prior},
        #'All wo prior': {'matrices':all_wo_prior_m, 'graphs': all_wo_prior},
        #'Original': {'matrices':original_m, 'graphs': original},
        'Complete': {'matrices':removed_nan_m, 'graphs': removed_nan},
        'Imputed': {'matrices':imputed_all_m, 'graphs': imputed_all},
        #'Imputed all': {'matrices':imputed_all_m, 'graphs': imputed_all},
        #'Imputed mixed': {'matrices':imputed_mixed_m, 'graphs': imputed_mixed},
        #'Imputed discrete': {'matrices':imputed_discrete_m, 'graphs': imputed_discrete}
    }
    if len(node_labels)<8:
        cg_pc_fisherz = digraph_to_binary_adj(Outputs_tumor_data['Data_imputed']['PC_fisherz'])
    Results = {}
    pos = None
    
    for strategy, c_graphs in Dict_adj_matrices.items():
        print(f"\nVisualizing aggregate adjacency for strategy: {strategy}")
        results = analyze_graph_stability_and_direction(c_graphs['graphs'])
        Results[strategy] = results
        pos = visualize_aggregate_adjacencies(c_graphs['matrices'], strategy, node_labels, pos, save_plots=True, out_dir=out_dir, show_plots=False)

    # Aggregate the computed Results and produce summary visualizations
    summary_df = aggregate_and_visualize_results(Results, node_labels=node_labels, save_plots=True, out_dir=out_dir, show_plots=False)
    print('\nAggregated summary table:')

    print(summary_df.groupby('strategy').agg({'n_edges':['mean','std'],'skeleton density':['mean','std'],'directionality index':['mean','std']}))

