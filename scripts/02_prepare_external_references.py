from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _resolve_ns3_lena_root() -> Path | None:
    env_value = os.environ.get("NS3_LENA_ROOT", "").strip()
    if not env_value:
        return None
    return Path(env_value).expanduser()


def main() -> None:
    from network_degradation_mining.config import load_paths, resolve_project_path
    from network_degradation_mining.external_reference import (
        prepare_external_references,
    )

    paths = load_paths(ROOT / "config" / "paths.yaml")
    ns3_lena_root = _resolve_ns3_lena_root()

    vienna_path = resolve_project_path(paths["raw"]["vienna_4g5g"], ROOT)
    campus_path = resolve_project_path(paths["raw"]["campus_qos"], ROOT)
    ucc_path = resolve_project_path(paths["raw"]["ucc_5g_context"], ROOT)
    simulator_reference_dir = resolve_project_path(
        paths["raw"]["simulator_reference"], ROOT
    )

    output_dir = resolve_project_path(paths["interim"]["external_reference_dir"], ROOT)
    mining_tables_dir = resolve_project_path(
        paths["processed"]["mining_tables_dir"], ROOT
    )
    results_dir = resolve_project_path(paths["results"]["external_reference_dir"], ROOT)

    output_paths = prepare_external_references(
        vienna_path=vienna_path,
        campus_path=campus_path,
        ucc_path=ucc_path,
        simulator_reference_dir=simulator_reference_dir,
        output_dir=output_dir,
        mining_tables_dir=mining_tables_dir,
        results_dir=results_dir,
        root=ROOT,
        ns3_lena_root=ns3_lena_root,
    )

    print("External reference preparation complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
