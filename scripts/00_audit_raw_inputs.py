from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_degradation_mining.config import load_dataset_registry
from network_degradation_mining.validation import audit_raw_inputs


def main() -> None:
    registry = load_dataset_registry(ROOT / "config" / "dataset_registry.yaml")
    output_paths = audit_raw_inputs(registry=registry, root=ROOT)

    print("Raw input audit complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()