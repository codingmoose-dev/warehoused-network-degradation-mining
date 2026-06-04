from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.association_rules import (
        run_association_rule_mining,
    )
    from network_degradation_mining.config import load_paths, resolve_project_path

    paths = load_paths(ROOT / "config" / "paths.yaml")

    association_table_path = resolve_project_path(
        Path(paths["processed"]["mining_tables_dir"])
        / "association_transaction_table.csv",
        ROOT,
    )
    results_dir = resolve_project_path(paths["results"]["association_rules_dir"], ROOT)
    figures_dir = resolve_project_path(paths["figures"]["association_rules_dir"], ROOT)

    output_paths = run_association_rule_mining(
        association_table_path=association_table_path,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    print("Association rule mining complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()