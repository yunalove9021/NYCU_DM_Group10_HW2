"""Dataset summaries reusable across experiments."""

from pathlib import Path

import pandas as pd


def summarize_labels(data_sets, output_file, label_column="y"):
    """Write label counts for (dataframe, name) pairs and return the table."""
    rows = []
    for data, name in data_sets:
        labels = data.get(label_column, pd.Series(float("nan"), index=data.index))
        rows.append({
            "dataset": name,
            "total": len(data),
            "labeled": int(labels.notna().sum()),
            "normal_0": int((labels == 0).sum()),
            "anomaly_1": int((labels == 1).sum()),
            "unlabeled_nan": int(labels.isna().sum()),
        })
    summary = pd.DataFrame(rows)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_excel(output_file, index=False)
    return summary
