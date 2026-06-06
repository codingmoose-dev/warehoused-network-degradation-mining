from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.graph_mining import run_graph_mining

    paths = load_paths(ROOT / "config" / "paths.yaml")

    graph_nodes_path = resolve_project_path(
        Path(paths["processed"]["graph_nodes_dir"]) / "all_nodes.csv",
        ROOT,
    )
    graph_edges_path = resolve_project_path(
        Path(paths["processed"]["graph_edges_dir"]) / "all_edges.csv",
        ROOT,
    )
    results_dir = resolve_project_path(paths["results"]["graph_dir"], ROOT)
    figures_dir = resolve_project_path(paths["figures"]["graph_dir"], ROOT)

    output_paths = run_graph_mining(
        graph_nodes_path=graph_nodes_path,
        graph_edges_path=graph_edges_path,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    print("Graph mining complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
