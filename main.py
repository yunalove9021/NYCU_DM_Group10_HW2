"""Run an experiment using an external YAML configuration."""

import argparse
from pathlib import Path

from src.config import load_config
from src.model.registry import MODEL_REGISTRY, get_model_runner, validate_model_name


def model_choice(value):
    try:
        return validate_model_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "configs" / "base.yaml",
        help="Path to the experiment YAML file",
    )
    parser.add_argument("--model", type=model_choice, default="RF",
                        metavar="{" + ",".join(MODEL_REGISTRY) + "}",
                        help="Run one independent model pipeline (default: RF)")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.model)
    run = get_model_runner(config["model"]["name"])
    return run(config)


if __name__ == "__main__":
    main()
