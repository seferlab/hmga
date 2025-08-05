
import networkx as nx
import numpy as np
import random
from typing import Tuple, List

def generate_napabench_like_graphs(
    num_nodes_ancestor: int = 2000,
    num_nodes_source: int = 3000,
    num_nodes_target: int = 4000,
    duplication_prob: float = 0.2,
    mutation_prob: float = 0.1,
    seed: int = 42
) -> Tuple[nx.Graph, nx.Graph, nx.Graph, List[Tuple[int, int]]]:
    random.seed(seed)
    np.random.seed(seed)
    ancestor = nx.barabasi_albert_graph(num_nodes_ancestor, 5)
    ancestor = nx.convert_node_labels_to_integers(ancestor)

    def evolve_graph(base_graph: nx.Graph, target_size: int) -> Tuple[nx.Graph, dict]:
        evolved = base_graph.copy()
        mapping = {n: n for n in evolved.nodes()}
        current_max = max(evolved.nodes())

        while len(evolved.nodes()) < target_size:
            node_to_duplicate = random.choice(list(base_graph.nodes()))
            current_max += 1
            evolved.add_node(current_max)
            neighbors = list(base_graph.neighbors(node_to_duplicate))
            for neighbor in neighbors:
                if random.random() > mutation_prob:
                    evolved.add_edge(current_max, mapping[neighbor])
            for neighbor in list(evolved.neighbors(current_max)):
                if random.random() < mutation_prob:
                    evolved.remove_edge(current_max, neighbor)
            mapping[node_to_duplicate] = current_max
        return evolved, mapping

    source_graph, source_map = evolve_graph(ancestor, num_nodes_source)
    target_graph, target_map = evolve_graph(ancestor, num_nodes_target)

    anchor_links = []
    for ancestor_node in ancestor.nodes():
        if ancestor_node in source_map and ancestor_node in target_map:
            anchor_links.append((source_map[ancestor_node], target_map[ancestor_node]))

    return ancestor, source_graph, target_graph, anchor_links
