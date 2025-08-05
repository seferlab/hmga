
import networkx as nx
import pandas as pd
from typing import Tuple, List

def load_graph_from_file(path: str, format: str = "edgelist") -> nx.Graph:
    if format == "edgelist":
        return nx.read_edgelist(path, nodetype=int)
    elif format == "graphml":
        return nx.read_graphml(path)
    elif format == "gml":
        return nx.read_gml(path)
    elif format == "csv":
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
            return G
        else:
            raise ValueError("CSV file must have at least two columns for edges.")
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_anchor_links(path: str) -> List[Tuple[int, int]]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Anchor file must have at least two columns")
    return list(zip(df.iloc[:, 0].astype(int), df.iloc[:, 1].astype(int)))

def load_real_graphs(
    source_graph_path: str,
    target_graph_path: str,
    anchor_links_path: str,
    fmt: str = "edgelist"
) -> Tuple[nx.Graph, nx.Graph, List[Tuple[int, int]]]:
    Gs = load_graph_from_file(source_graph_path, format=fmt)
    Gt = load_graph_from_file(target_graph_path, format=fmt)
    anchors = load_anchor_links(anchor_links_path)
    return Gs, Gt, anchors
