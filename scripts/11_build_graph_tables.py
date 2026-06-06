from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.graph_builder import build_graph_tables

    paths = load_paths(ROOT / "config" / "paths.yaml")

    warehouse_dir = resolve_project_path(paths["processed"]["warehouse_dir"], ROOT)
    graph_nodes_dir = resolve_project_path(paths["processed"]["graph_nodes_dir"], ROOT)
    graph_edges_dir = resolve_project_path(paths["processed"]["graph_edges_dir"], ROOT)
    results_dir = resolve_project_path(paths["results"]["graph_dir"], ROOT)
    external_reference_dir = resolve_project_path(
        paths["interim"]["external_reference_dir"],
        ROOT,
    )
    external_results_dir = resolve_project_path(
        paths["results"]["external_reference_dir"],
        ROOT,
    )

    output_paths = build_graph_tables(
        warehouse_dir=warehouse_dir,
        graph_nodes_dir=graph_nodes_dir,
        graph_edges_dir=graph_edges_dir,
        results_dir=results_dir,
        external_reference_dir=external_reference_dir,
        external_results_dir=external_results_dir,
        project_root_dir=ROOT,
    )

    print("Graph table build complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
