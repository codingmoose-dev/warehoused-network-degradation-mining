from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.features import build_mining_feature_tables

    paths = load_paths(ROOT / "config" / "paths.yaml")

    warehouse_dir = resolve_project_path(paths["processed"]["warehouse_dir"], ROOT)
    mining_tables_dir = resolve_project_path(
        paths["processed"]["mining_tables_dir"], ROOT
    )
    results_dir = resolve_project_path(
        paths["results"].get("audits_dir", "results/audits/"), ROOT
    )

    output_paths = build_mining_feature_tables(
        warehouse_dir=warehouse_dir,
        mining_tables_dir=mining_tables_dir,
        results_dir=results_dir,
    )

    print("Mining feature table build complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
