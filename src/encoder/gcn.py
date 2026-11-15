"""Supervised two-layer GCN with a reusable 100-dimensional node encoder."""

from copy import deepcopy
import random
from time import perf_counter

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler

from src.utils.graph import edge_indices
from src.utils.run_log import synchronize_device


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=100, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=False)
        self.conv2 = GCNConv(hidden_dim, embedding_dim, cached=False)
        self.classifier = nn.Linear(embedding_dim, 2)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

    def forward(self, x, edge_index):
        return self.classifier(self.encode(x, edge_index))


def fit_encoder(datasets, edges, features, label, settings, seed, run_log=None):
    """Fit on train labels only; select checkpoint using validation loss."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = settings.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    params = settings["params"]
    scaler = StandardScaler()
    scaler.fit(datasets["train"][features])
    undirected = settings.get("undirected", False)
    graphs = {}
    # Test data is deliberately excluded from optimization and checkpoint selection.
    for name in ("train", "val"):
        data = datasets[name]
        x = (scaler.transform(data[features]).astype(np.float32) if len(data)
             else np.empty((0, len(features)), dtype=np.float32))
        if not np.isfinite(x).all():
            raise ValueError(f"{name}: features contain NaN or Inf")
        y = data.get(label)
        if y is None:
            y = np.full(len(data), np.nan)
        y = np.asarray(y, dtype=float)
        known = np.isfinite(y)
        if not np.isin(y[known], [0, 1]).all():
            raise ValueError(f"{name}: labels must be binary or missing")
        graphs[name] = (
            torch.as_tensor(x, device=device),
            torch.as_tensor(edge_indices(data, edges[name], undirected), device=device),
            torch.as_tensor(np.where(known, y, 0), dtype=torch.long, device=device),
            torch.as_tensor(known, device=device),
        )
    x, edge_index, y, mask = graphs["train"]
    counts = torch.bincount(y[mask], minlength=2)
    if (counts == 0).any():
        raise ValueError("GCN training requires labeled examples of both classes")
    weights = mask.sum() / (2 * counts.float())
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = GCNEncoder(len(features), params["hidden_dim"], params["embedding_dim"],
                       params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    history, best_state, best_loss, best_epoch, stale = [], None, float("inf"), 0, 0
    val_x, val_edges, val_y, val_mask = graphs["val"]
    has_validation = bool(val_mask.any())
    for epoch in range(1, params["epochs"] + 1):
        synchronize_device(device)
        epoch_started = perf_counter()
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x, edge_index)[mask], y[mask])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite GCN training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x, val_edges)[val_mask], val_y[val_mask])) if has_validation else None
        if val_loss is not None and not np.isfinite(val_loss):
            raise ValueError("Nonfinite GCN validation loss")
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_loss": val_loss})
        # Without validation labels, use the final epoch and disable early stopping.
        if not has_validation or val_loss < best_loss:
            best_state, best_epoch, stale = deepcopy(model.state_dict()), epoch, 0
            best_loss = val_loss if has_validation else float(loss.detach())
        else:
            stale += 1
        print(f"GCN seed={seed} epoch={epoch}: loss={float(loss.detach()):.6f}, val_loss={val_loss}")
        synchronize_device(device)
        epoch_seconds = perf_counter() - epoch_started
        history[-1]["elapsed_seconds"] = epoch_seconds
        if run_log is not None:
            run_log.event("gcn_epoch", epoch=epoch, train_loss=history[-1]["train_loss"],
                          val_loss=val_loss, elapsed_seconds=epoch_seconds, best_epoch=best_epoch,
                          stale_epochs=stale, device=str(device))
        if has_validation and stale >= params["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = {"seed": seed, "best_epoch": best_epoch, "device": str(device),
              "class_weights": weights.cpu().tolist(), "history": history,
              "checkpoint_metric": "val_loss" if has_validation else "last_epoch"}
    return model, scaler, report


def encode_nodes(model, scaler, data, edges, features, undirected=False):
    """Return embeddings in exactly the input node row order."""
    device = next(model.parameters()).device
    if not len(data):
        edge_indices(data, edges, undirected)
        return np.empty((0, model.conv2.out_channels), dtype=np.float32)
    x = scaler.transform(data[features]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features contain NaN or Inf")
    model.eval()
    with torch.no_grad():
        result = model.encode(torch.as_tensor(x, device=device),
                              torch.as_tensor(edge_indices(data, edges, undirected), device=device))
    return result.cpu().numpy()


def load_encoder(checkpoint_file, scaler_file, device="cpu"):
    """Restore a saved encoder and its train-fitted scaler for new graphs."""
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    params = checkpoint["settings"]["params"]
    model = GCNEncoder(len(checkpoint["input_features"]), params["hidden_dim"],
                       params["embedding_dim"], params["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, joblib.load(scaler_file), checkpoint
