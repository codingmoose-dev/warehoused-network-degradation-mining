from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.gnn import run_graph_neural_benchmark

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
    supervised_model_comparison_path = resolve_project_path(
        Path(paths["results"]["supervised_dir"]) / "model_comparison.csv",
        ROOT,
    )

    output_paths = run_graph_neural_benchmark(
        graph_nodes_path=graph_nodes_path,
        graph_edges_path=graph_edges_path,
        results_dir=results_dir,
        figures_dir=figures_dir,
        supervised_model_comparison_path=supervised_model_comparison_path,
    )

    print("Graph neural benchmark complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
