from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.olap import run_descriptive_warehouse_mining

    paths = load_paths(ROOT / "config" / "paths.yaml")

    warehouse_dir = resolve_project_path(paths["processed"]["warehouse_dir"], ROOT)
    results_dir = resolve_project_path(paths["results"]["descriptive_mining_dir"], ROOT)
    figures_dir = resolve_project_path(paths["figures"]["descriptive_mining_dir"], ROOT)

    output_paths = run_descriptive_warehouse_mining(
        warehouse_dir=warehouse_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    print("Descriptive warehouse mining complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
