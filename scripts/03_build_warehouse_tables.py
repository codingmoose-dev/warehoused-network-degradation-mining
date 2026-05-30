from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_degradation_mining.config import load_paths, resolve_project_path
from network_degradation_mining.warehouse import build_warehouse_tables


def main() -> None:
    paths = load_paths(ROOT / "config" / "paths.yaml")

    input_path = resolve_project_path(paths["interim"]["synnetqos_core"], ROOT)
    warehouse_dir = resolve_project_path(paths["processed"]["warehouse_dir"], ROOT)
    results_dir = resolve_project_path(paths["results"]["warehouse_dir"], ROOT)
    output_paths = build_warehouse_tables(
        input_path=input_path,
        warehouse_dir=warehouse_dir,
        results_dir=results_dir,
    )

    print("Warehouse table build complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()