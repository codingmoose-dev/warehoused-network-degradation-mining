from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.decision_tree_rules import (
        run_decision_tree_rule_extraction,
    )

    paths = load_paths(ROOT / "config" / "paths.yaml")

    supervised_table_path = resolve_project_path(
        Path(paths["processed"]["mining_tables_dir"])
        / "supervised_degradation_table.csv",
        ROOT,
    )
    results_dir = resolve_project_path(
        paths["results"]["decision_tree_rules_dir"],
        ROOT,
    )
    figures_dir = resolve_project_path(paths["figures"]["decision_tree_dir"], ROOT)

    output_paths = run_decision_tree_rule_extraction(
        supervised_table_path=supervised_table_path,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    print("Decision-tree rule extraction complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
