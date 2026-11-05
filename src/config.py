"""Load and validate experiment settings before creating artifacts."""

from pathlib import Path
import json
import re

from src.model.registry import validate_model_name


def resolve_feature_set(data, config_dir):
    """Resolve JSON feature lists relative to YAML and set the output name."""
    features = data.get("feature_columns")
    if isinstance(features, str):
        feature_path = (Path(config_dir) / features).resolve()
        with feature_path.open(encoding="utf-8-sig") as stream:
            features = json.load(stream)
        set_name = feature_path.stem
        data["feature_source"] = feature_path
    else:
        set_name = data.get("set_name", "basic" if features is None else "custom")
    if features is not None and (
        not isinstance(features, list) or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(features) != len(set(features))
        or data.get("label_column", "y") in features
    ):
        raise ValueError("feature_columns must contain a nonempty list of unique feature names excluding the label")
    if isinstance(data.get("feature_columns"), str) and features is None:
        raise ValueError("Feature JSON must contain a nonempty list of feature names")
    if not isinstance(set_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", set_name):
        raise ValueError("set_name must contain only letters, digits, underscores, or hyphens")
    data["feature_columns"] = features
    data["set_name"] = set_name


def resolve_feature_sets(data, config_dir):
    """Normalize one or more JSON paths into named feature sets."""
    columns = data.get("feature_columns")
    if isinstance(columns, list) and any(
        isinstance(value, str) and value.lower().endswith(".json") for value in columns
    ):
        if not all(isinstance(value, str) and value.lower().endswith(".json") for value in columns):
            raise ValueError("feature_columns cannot mix JSON paths and inline column names")
        specifications = columns
    else:
        specifications = [columns]
    sets = []
    names = set()
    for specification in specifications:
        feature_set = {"feature_columns": specification,
                       "label_column": data.get("label_column", "y")}
        if "set_name" in data:
            feature_set["set_name"] = data["set_name"]
        resolve_feature_set(feature_set, config_dir)
        name = feature_set["set_name"]
        if name in names:
            raise ValueError(f"Duplicate feature set name: {name}")
        names.add(name)
        sets.append({"set_name": name, "feature_columns": feature_set["feature_columns"],
                     "feature_source": feature_set.get("feature_source")})
    data["feature_sets"] = sets


def load_config(path, model_name="RF"):
    validate_model_name(model_name)
    import yaml

    path = Path(path).resolve()
    # All selected models update a single index beside the YAML configuration.
    config_index = path.parent / "model.json"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config["model_index"] = config_index
    for key in ("input", "output", "random_seeds", "data", "models"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    for key in ("input", "output"):
        config[key] = (path.parent / config[key]).resolve()
    seeds = config["random_seeds"]
    if isinstance(seeds, int) and not isinstance(seeds, bool):
        seeds = [seeds]
    if not isinstance(seeds, list) or not seeds or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        raise ValueError("random_seeds must be an integer or nonempty list of uint32 integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds must not contain duplicates")
    config["random_seeds"] = seeds
    models = config["models"]
    if not isinstance(models, dict) or not isinstance(models.get(model_name), dict):
        raise ValueError(f"Missing model settings: models.{model_name}")
    model = {**models[model_name], "name": model_name}
    config["model"] = model
    params = model.setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"models.{model_name}.params must be a mapping")
    if "random_state" in params:
        raise ValueError(f"Set random_seeds instead of models.{model_name}.params.random_state")
    data = config["data"]
    resolve_feature_sets(data, path.parent)
    for key in ("train_file", "val_file", "test_files"):
        if key not in data:
            raise ValueError(f"Missing data.{key}")
    tests = data["test_files"]
    if isinstance(tests, str):
        tests = [tests]
    if isinstance(tests, list):
        names = [Path(filename).stem for filename in tests]
        if len(names) != len(set(names)):
            raise ValueError("Test filenames must have unique stems; use named mapping instead")
        tests = dict(zip(names, tests))
    if not isinstance(tests, dict):
        raise ValueError("data.test_files must be a mapping, list, or filename")
    for name in tests:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in {"train", "val"}:
            raise ValueError(f"Invalid or reserved test dataset name: {name}")
    data["test_files"] = tests
    for filename in (data["train_file"], data["val_file"], *tests.values()):
        file_path = config["input"] / filename
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
    if model["name"] == "GCN_RF":
        validate_gcn_config(config)
    return config


def validate_gcn_config(config):
    data = config["data"]
    edges = data.get("edge_files", {})
    expected = {"train", "val", *data["test_files"]}
    if not isinstance(edges, dict) or set(edges) != expected:
        raise ValueError("data.edge_files must match train, val and all named test datasets")
    for filename in edges.values():
        if not (config["input"] / filename).is_file():
            raise FileNotFoundError(config["input"] / filename)
    encoder = config["model"].get("encoder", {})
    if encoder.get("name") != "GCN":
        raise ValueError("GCN_RF requires models.GCN_RF.encoder.name: GCN")
    params = encoder.get("params", {})
    required = {"hidden_dim", "embedding_dim", "num_layers", "dropout",
                "learning_rate", "epochs", "patience"}
    if not isinstance(params, dict) or not required <= params.keys():
        raise ValueError(f"encoder.params requires {sorted(required)}")
    for key in ("hidden_dim", "embedding_dim", "epochs", "patience"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"encoder.params.{key} must be a positive integer")
    if params["num_layers"] != 2 or params["embedding_dim"] != 100:
        raise ValueError("GCN_RF requires 2 GCN layers and 100 embedding dimensions")
    if not 0 <= params["dropout"] < 1 or params["learning_rate"] <= 0:
        raise ValueError("Invalid encoder dropout or learning_rate")
    if encoder.get("loss") != {"name": "weighted_cross_entropy", "class_weights": "balanced"}:
        raise ValueError("Use weighted_cross_entropy with balanced class_weights")
    if encoder.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("encoder.device must be auto, cpu, or cuda")
    if type(encoder.get("undirected", False)) is not bool:
        raise ValueError("encoder.undirected must be a boolean")
