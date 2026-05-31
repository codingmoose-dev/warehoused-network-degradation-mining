from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.preprocessing import prepare_synnetqos_core

    paths = load_paths(ROOT / "config" / "paths.yaml")

    input_path = resolve_project_path(paths["raw"]["synnetqos"], ROOT)
    output_path = resolve_project_path(paths["interim"]["synnetqos_core"], ROOT)
    audit_dir = resolve_project_path(paths["results"]["audits_dir"], ROOT)

    output_paths = prepare_synnetqos_core(
        input_path=input_path,
        output_path=output_path,
        audit_dir=audit_dir,
    )

    print("SynNetQoS core preparation complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
