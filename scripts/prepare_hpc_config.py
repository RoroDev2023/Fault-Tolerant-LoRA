"""Create a temporary cluster-specific config without changing a checked-in config."""

import argparse
import json
from pathlib import Path


def create_overlay(input_path, output_path, *, device=None, data_root=None):
    config = json.loads(Path(input_path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {input_path}")

    if device is not None:
        if "device" not in config:
            raise ValueError(f"Config has no device setting to override: {input_path}")
        config["device"] = device
    if data_root is not None:
        if "data_root" not in config:
            raise ValueError(f"Config has no data_root setting to override: {input_path}")
        config["data_root"] = str(data_root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(output_path)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="checked-in experiment JSON config")
    parser.add_argument("--output", type=Path, required=True, help="path for the derived config")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    try:
        create_overlay(args.input, args.output, device=args.device, data_root=args.data_root)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote derived config: {args.output}")


if __name__ == "__main__":
    main()
