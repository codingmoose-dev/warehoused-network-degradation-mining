from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.clustering import run_clustering_analysis
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.plotting import plot_clustering_analysis_outputs

    paths = load_paths(ROOT / "config" / "paths.yaml")

    clustering_table_path = resolve_project_path(
        Path(paths["processed"]["mining_tables_dir"]) / "clustering_feature_table.csv",
        ROOT,
    )
    results_dir = resolve_project_path(paths["results"]["clustering_dir"], ROOT)
    figures_dir = resolve_project_path(paths["figures"]["clustering_dir"], ROOT)

    output_paths = run_clustering_analysis(
        clustering_table_path=clustering_table_path,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )
    output_paths.update(
        plot_clustering_analysis_outputs(
            results_dir=results_dir,
            figures_dir=figures_dir,
        )
    )

    print("Clustering analysis complete.")
    for name, path in sorted(output_paths.items()):
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
