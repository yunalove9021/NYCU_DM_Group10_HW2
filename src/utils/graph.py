"""Map transaction IDs to graph indices without dropping invalid edges."""

import numpy as np
import pandas as pd


def edge_indices(nodes, edges, undirected=False):
    if "txId" not in nodes or nodes["txId"].isna().any() or nodes["txId"].duplicated().any():
        raise ValueError("Nodes require unique nonmissing txId values")
    if not {"txId1", "txId2"} <= set(edges.columns):
        raise ValueError("Edges require txId1 and txId2")
    indices = pd.Series(np.arange(len(nodes)), index=nodes["txId"])
    source, target = edges["txId1"].map(indices), edges["txId2"].map(indices)
    if source.isna().any() or target.isna().any():
        raise ValueError("Edge endpoints must exist in the corresponding node dataset")
    result = np.stack([source.to_numpy(dtype=np.int64), target.to_numpy(dtype=np.int64)])
    if undirected:
        result = np.concatenate([result, result[::-1]], axis=1)
    return np.unique(result, axis=1)
