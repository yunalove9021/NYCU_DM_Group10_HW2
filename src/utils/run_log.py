"""Per-run training/prediction events and wall-clock timings (Taipei timestamps)."""

from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
TAIPEI = timezone(timedelta(hours=8))
TIMING_COLUMNS = ["timestamp", "run_id", "model_name", "feature_set", "seed", "operation",
                  "stage", "dataset", "status", "elapsed_seconds", "n_rows", "n_features",
                  "device", "error_type", "error"]


def synchronize_device(device):
    """Wait for GPU work without importing torch for a pure RF run."""
    if device is not None and str(device).startswith("cuda"):
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(device)


class RunLog:
    def __init__(self, model_name, feature_set, seed, operation="train", directory=None):
        for value in (model_name, feature_set, operation):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"Invalid log identifier: {value}")
        now = datetime.now(TAIPEI)
        self.context = {"run_id": now.strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8],
                        "model_name": model_name, "feature_set": feature_set,
                        "seed": int(seed), "operation": operation}
        folder = (Path(directory) if directory is not None else ROOT / "Analysis" / "log") / model_name / feature_set
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{feature_set}_seed_{seed}_{model_name}_{operation}_{self.context['run_id']}"
        self.path = folder / f"{prefix}.log"
        self.timing_path = folder / f"{prefix}_timings.csv"
        self.event("run_created", timing_file=str(self.timing_path.resolve()))

    def event(self, event, **fields):
        record = {**self.context, **fields, "event": event,
                  "timestamp": datetime.now(TAIPEI).isoformat(timespec="milliseconds")}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        return record

    @contextmanager
    def stage(self, stage, dataset=None, device=None, **fields):
        """Time a stage, including GPU completion; preserve exceptions in the log."""
        details = {"stage": stage, "dataset": dataset, "device": str(device) if device is not None else None, **fields}
        self.event("stage_start", **details)
        synchronize_device(device)
        started = perf_counter()
        try:
            yield details
            synchronize_device(device)
        except BaseException as error:
            self._finish(details, perf_counter() - started, "failed", error)
            raise
        else:
            self._finish(details, perf_counter() - started, "ok")

    def _finish(self, details, elapsed, status, error=None):
        fields = {**details, "status": status, "elapsed_seconds": elapsed}
        if error is not None:
            fields.update(error_type=type(error).__name__, error=str(error))
        record = self.event("stage_end", **fields)
        new_file = not self.timing_path.exists()
        with self.timing_path.open("a", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIMING_COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(record)
