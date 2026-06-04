from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.external_comparison import (
        run_external_reference_comparison,
    )

    paths = load_paths(ROOT / "config" / "paths.yaml")

    synnetqos_core_path = resolve_project_path(paths["interim"]["synnetqos_core"], ROOT)
    external_reference_dir = resolve_project_path(
        paths["interim"]["external_reference_dir"], ROOT
    )
    results_dir = resolve_project_path(paths["results"]["external_reference_dir"], ROOT)
    figures_dir = resolve_project_path(paths["figures"]["external_reference_dir"], ROOT)

    output_paths = run_external_reference_comparison(
        synnetqos_core_path=synnetqos_core_path,
        external_reference_dir=external_reference_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    print("External reference comparison complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
