import pandas as pd
from typing import Tuple, Dict, List
import networkx as nx
import numpy as np

import warnings # To suppress some warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="causallearn")

# Causal-learn imports
try:
    from causallearn.search.ConstraintBased.PC import pc as PC 
    from causallearn.search.ConstraintBased.FCI import fci as FCI 
    from causallearn.search.ScoreBased.GES import ges as GES 
    from causallearn.search.FCMBased import lingam
    from causallearn.graph.Graph import Graph as CausalLearnGraph 
    from causallearn.utils.cit import CIT, fisherz, gsq, chisq, kci, mv_fisherz
    from causallearn.score.LocalScoreFunction import local_score_BIC, local_score_BDeu, local_score_cv_general, local_score_marginal_general
    from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.GraphUtils import GraphUtils
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.search.PermutationBased.GRaSP import grasp
    from causallearn.search.PermutationBased.BOSS import boss
    _CAUSAL_LEARN_AVAILABLE = True
except ImportError as e:
    print(f"Error: causallearn library or its components not found ({e}).")
    _CAUSAL_LEARN_AVAILABLE = False
    exit(1)

# Extract adjacency matrix based on expected structure
def _causal_learn_graph_to_networkx(cl_graph_object, num_nodes: int) -> nx.DiGraph:
    """
    Converts a causallearn graph object to a networkx.DiGraph.
    Interprets edge types based on the detected causallearn adjacency matrix convention
    and adds appropriate 'edge_type' attributes to the NetworkX graph.
    """
    adj_matrix = None
    graph_type_convention = None # 'PC_TYPE' or 'FCI_TYPE'

    # Determine how to get the adjacency matrix and identify graph type convention
    if hasattr(cl_graph_object, 'G') and hasattr(cl_graph_object.G, 'graph'):
        adj_matrix = cl_graph_object.G.graph
    elif hasattr(cl_graph_object, 'graph'):
        adj_matrix = cl_graph_object.graph
    else:
        raise TypeError(f"Unexpected graph object structure from causallearn: {type(cl_graph_object)}. Cannot extract adjacency matrix.")

    if adj_matrix is None or adj_matrix.shape[0] != num_nodes or adj_matrix.shape[1] != num_nodes:
        raise ValueError("Invalid or unextractable adjacency matrix from causallearn graph object.")

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    processed_pairs = set() # To avoid redundant processing for undirected/bidirected pairs

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j or (j, i) in processed_pairs: # Skip self-loops and already processed symmetric pairs
                continue

            val_ij = adj_matrix[i, j]
            val_ji = adj_matrix[j, i]

            if (val_ji == 1 and val_ij == -1) or \
                (val_ij == 1 and val_ji == 0) or \
                    (val_ij == 2 and val_ji == 1): # FCI convention
                G.add_edge(i, j, edge_type='directed') # i -> j
                processed_pairs.add((i, j)) # Mark as processed in both directions
            elif (val_ji == -1 and val_ij == 1) or \
                (val_ij == 0 and val_ji == 1) or \
                    (val_ij == 1 and val_ji == 2): # FCI convention
                G.add_edge(j, i, edge_type='directed') # j -> i
                processed_pairs.add((i, j)) # Mark as processed in both directions

            elif val_ij == -1 and val_ji == -1 or \
                (val_ij == 2 and val_ji == 2): # i -- j (undirected) - PC and GES (Record['G'] type) convention; FCI convention
                G.add_edge(i, j, edge_type='undirected')
                G.add_edge(j, i, edge_type='undirected')
                processed_pairs.add((i, j)) # Mark as processed in both directions
            elif val_ij == 1 and val_ji == 1: # i <-> j (bidirected) - PC and FCI convention
                G.add_edge(i, j, edge_type='bidirected')
                G.add_edge(j, i, edge_type='bidirected')
                processed_pairs.add((i, j)) # Mark as processed in both directions

    return G

# --- Background Knowledge for Causal Discovery ---
# PC format
def prior(num_nodes=7) -> BackgroundKnowledge:
    cg = CausalGraph(num_nodes)
    nodes = cg.G.get_nodes()
    bk=BackgroundKnowledge()

    for i in range(num_nodes):
        if i!=0: bk.add_forbidden_by_node(nodes[i], nodes[0]) #Age has no parents
        if i!=1: bk.add_forbidden_by_node(nodes[i], nodes[1]) #Sex has no parents
        if i!=5: bk.add_forbidden_by_node(nodes[5], nodes[i]) #Status has no children

    return bk

def prior_knowledge(num_nodes=7) -> np.array:
    #DirectLiNGAM format
    prior_knowledge = np.ones((num_nodes,num_nodes))*(-1)
    for i in range(num_nodes):
        if i!=0: prior_knowledge[i, 0] = 0 #Age no parents
        if i!=1: prior_knowledge[i, 1] = 0 #Sex no parents
        if i!=5: prior_knowledge[5, i] = 0 #Status no children
        
    return prior_knowledge

def directlingam_discovery(data: np.array, priorknowledge: np.array = None) -> nx.DiGraph:
    """DirectLiNGAM discovery (continuous data only, via causallearn)."""
    out = None
    try:
        model = lingam.DirectLiNGAM(prior_knowledge=priorknowledge)
        model.fit(data)
        adj_matrix_T = model.adjacency_matrix_
        G = nx.DiGraph()
        G.add_nodes_from(range(adj_matrix_T.shape[0]))
        for i in range(adj_matrix_T.shape[0]):
            for j in range(adj_matrix_T.shape[1]):
                if adj_matrix_T[i, j] != 0:
                    G.add_edge(j, i, edge_type='directed')
        out = G
        return out
    except Exception as e:
        print(f"Warning: DirectLiNGAM failed: {e}.")
        pass

def rcd_discovery(data: np.array, max_explanatory_num: int = 2, cor_alpha: float = 0.01, ind_alpha: float = 0.01, shapiro_alpha: float = 0.01) -> nx.DiGraph:
    """RCD discovery (continuous data only, via causallearn). LiNGAM with latent variables"""
    out = None
    try:
        model = lingam.RCD(max_explanatory_num, cor_alpha, ind_alpha, shapiro_alpha)
        model.fit(data)
        adj_matrix_T = model.adjacency_matrix_
        G = nx.DiGraph()
        G.add_nodes_from(range(adj_matrix_T.shape[0]))
        for i in range(adj_matrix_T.shape[0]):
            for j in range(adj_matrix_T.shape[1]):
                if adj_matrix_T[i, j] != 0:
                    G.add_edge(j, i, edge_type='directed')
        out = G
        return out
    except Exception as e:
        print(f"Warning: RCD-LiNGAM failed: {e}.")
        pass

def pc_discovery(data, indep_test = fisherz, uc_rule = 2, background_knowledge = None, mvpc=False) -> nx.DiGraph:
    """PC algorithm using causallearn, with fallback."""
    out = None
    try:
        graph_cl_result = PC(data, indep_test = indep_test, uc_rule = uc_rule, mvpc = mvpc, background_knowledge = background_knowledge) 
        # PC returns a CausalGraph object directly. Pass this object to the converter.
        out = _causal_learn_graph_to_networkx(graph_cl_result, data.shape[1])
        return out
    except Exception as e:
        print(f"Warning: PC algorithm failed: {e}. Returning simulated DAG.")
        pass

def fci_discovery(data, indep_test = fisherz, background_knowledge = None) -> nx.DiGraph:
    """FCI algorithm using causallearn."""
    out = None
    try:
        graph_cl_tuple, _ = FCI(data, indep_test=indep_test, background_knowledge=background_knowledge) 
        # FCI returns a tuple (GeneralGraph, edges). Pass the GeneralGraph object (index 0) to the converter.
        out = _causal_learn_graph_to_networkx(graph_cl_tuple, data.shape[1])
        return out
    except Exception as e:
        print(f"Warning: FCI algorithm failed: {e}.")
        pass

def ges_discovery(data, score_func="local_score_BIC") -> nx.DiGraph:
    """GES algorithm using causallearn."""
    out = None
    try:
        # GES: data, score_func
        graph_cl_result = GES(data, score_func=score_func) 
        # GES returns a GeneralGraph object directly. Pass this object to the converter.
        out = _causal_learn_graph_to_networkx(graph_cl_result['G'], data.shape[1])
        return out
    except Exception as e:
        print(f"Warning: GES algorithm failed: {e}.")
        pass
    
def exactsearch_discovery(data, search_method = 'astar') -> nx.DiGraph:
    """Exact search discovery (via causallearn)."""
    out = None
    try:
        adj_matrix, _ = bic_exact_search(data, search_method=search_method)
        G = nx.DiGraph()
        G.add_nodes_from(range(adj_matrix.shape[0]))
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] != 0:
                    G.add_edge(i, j, edge_type='directed')
        out = G
        return out
    except Exception as e:
        print(f"Warning: Exactsearch failed: {e}.")
        pass

def grasp_discovery(data, score_func="local_score_BIC") -> nx.DiGraph:
    """GraSP algorithm using causallearn."""
    out = None
    try:
        graph_cl_result = grasp(data, score_func=score_func) 
        # GraSP returns a GeneralGraph object directly. Pass this object to the converter.
        out = _causal_learn_graph_to_networkx(graph_cl_result, data.shape[1])
        return out
    except Exception as e:
        print(f"Warning: PC algorithm failed: {e}.")
        pass

def boss_discovery(data, score_func="local_score_BIC") -> nx.DiGraph:
    """BOSS algorithm using causallearn."""
    out = None
    try:
        graph_cl_result = boss(data, score_func=score_func) 
        # BOSS returns a GeneralGraph object directly. Pass this object to the converter.
        out = _causal_learn_graph_to_networkx(graph_cl_result, data.shape[1])
        return out
    except Exception as e:
        print(f"Warning: PC algorithm failed: {e}.")
        pass

# --- Causal Discovery Benchmarking Function ---
def causallearn_benchmark_(data: pd.DataFrame, is_discrete: bool, is_complete: bool, is_imputed: bool) -> Dict[str, callable]:
    """Returns a dictionary of causallearn-based causal discovery methods outputs (adjacency matrix)."""
    data_np = data.values
    num_nodes = data_np.shape[1]

    if is_discrete:
        Outuputs = {'Data_discrete': {}}
        out = pc_discovery(data=data_np, indep_test=chisq)
        if out is not None: Outuputs['Data_discrete']['PC_chisq'] = out
        out = pc_discovery(data=data_np, indep_test=chisq, background_knowledge=prior(num_nodes))
        if out is not None: Outuputs['Data_discrete']['PC_chisq_prior'] = out

        out = pc_discovery(data=data_np, indep_test=gsq)
        if out is not None: Outuputs['Data_discrete']['PC_gsq'] = out
        out = pc_discovery(data=data_np, indep_test=gsq, background_knowledge=prior(num_nodes))
        if out is not None: Outuputs['Data_discrete']['PC_gsq_prior'] = out

        out = fci_discovery(data=data_np, indep_test=chisq)
        if out is not None: Outuputs['Data_discrete']['FCI_chisq'] = out
        out = fci_discovery(data=data_np, indep_test=chisq, background_knowledge=prior(num_nodes))
        if out is not None: Outuputs['Data_discrete']['FCI_chisq_prior'] = out

        out = fci_discovery(data=data_np, indep_test=gsq)
        if out is not None: Outuputs['Data_discrete']['FCI_gsq'] = out
        out = fci_discovery(data=data_np, indep_test=gsq, background_knowledge=prior(num_nodes))
        if out is not None: Outuputs['Data_discrete']['FCI_gsq_prior'] = out
            
        out = ges_discovery(data=data_np, score_func='local_score_BDeu')
        if out is not None: Outuputs['Data_discrete']['GES_local_score_BDeu'] = out

        out = grasp_discovery(data=data_np, score_func='local_score_BDeu')
        if out is not None: Outuputs['Data_discrete']['GRaSP_local_score_BDeu'] = out

        out = boss_discovery(data=data_np, score_func='local_score_BDeu')
        if out is not None: Outuputs['Data_discrete']['BOSS_local_score_BDeu'] = out
    else:
        if not is_complete:
            Outuputs = {'Data_with_missing': {}}
            out = pc_discovery(data=data_np, indep_test=mv_fisherz, mvpc=True)
            if out is not None: Outuputs['Data_with_missing']['PC_mv_fisherz'] = out
            out = pc_discovery(data=data_np, indep_test=mv_fisherz, mvpc=True, background_knowledge=prior(num_nodes))
            if out is not None: Outuputs['Data_with_missing']['PC_mv_fisherz_prior'] = out

            out = fci_discovery(data=data_np, indep_test=mv_fisherz)
            if out is not None: Outuputs['Data_with_missing']['FCI_mv_fisherz'] = out
            out = fci_discovery(data=data_np, indep_test=mv_fisherz, background_knowledge=prior(num_nodes))
            if out is not None: Outuputs['Data_with_missing']['FCI_mv_fisherz_prior'] = out
        else:
            key = 'Data_imputed' if is_imputed else 'Data_no_nan'   
            Outuputs = {key: {}} 

            for indep in [fisherz, kci]:

                method_name = f'PC_{indep}'
                out = pc_discovery(data=data_np, indep_test=indep)
                if out is not None: Outuputs[key][method_name] = out
                out = pc_discovery(data=data_np, indep_test=indep, background_knowledge=prior(num_nodes))
                if out is not None: Outuputs[key][method_name+'_prior'] = out

                method_name = f'FCI_{indep}'
                out = fci_discovery(data=data_np, indep_test=indep)
                if out is not None: Outuputs[key][method_name] = out
                out = fci_discovery(data=data_np, indep_test=indep, background_knowledge=prior(num_nodes))
                if out is not None: Outuputs[key][method_name+'_prior'] = out

            for score in ['local_score_BIC','local_score_CV_general','local_score_marginal_general']:

                method_name = f'GES_{score}'
                out = ges_discovery(data=data_np, score_func=score)
                if out is not None: Outuputs[key][method_name] = out

            for search_method in ['astar','dp']:

                method_name = f'ExactBIC_{search_method}'
                out = exactsearch_discovery(data=data_np, search_method=search_method)
                if out is not None: Outuputs[key][method_name] = out

            for score in ['local_score_BIC','local_score_CV_general','local_score_marginal_general']:

                method_name = f'GRaSP_{score}'
                out = grasp_discovery(data=data_np, score_func=score)
                if out is not None: Outuputs[key][method_name] = out
            
            for score in ['local_score_BIC','local_score_CV_general','local_score_marginal_general']:

                method_name = f'BOSS_{score}'
                out = boss_discovery(data=data_np, score_func=score)
                if out is not None: Outuputs[key][method_name] = out

            out = directlingam_discovery(data=data_np)
            if out is not None: Outuputs[key]['DirectLiNGAM'] = out
            out = directlingam_discovery(data=data_np, priorknowledge=prior_knowledge(num_nodes))
            if out is not None: Outuputs[key]['DirectLiNGAM_prior'] = out

            

    return Outuputs
            
