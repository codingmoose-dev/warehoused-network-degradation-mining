# Warehouse-Guided Mining and Modeling of Network Service Degradation Patterns

This repository contains the code, configuration, tests, and generated outputs for a warehouse-guided study of network service-degradation patterns. The workflow transforms session-level network-performance records into an analytical warehouse, derives mining-ready tables, evaluates degradation models, compares selected variables against external reference sources, and constructs graph views for degradation-pattern analysis.

## Analytical workflow

The repository is organized into four analytical layers.

### Layer-1: Warehouse construction

Session-level network-performance records are transformed into a fact table and context dimensions. The warehouse separates measurement records from time, area, device, network, radio, application, mobility, environmental, and service-state context.

### Layer-2: Mining-table preparation

Warehouse outputs are converted into task-specific tables for descriptive mining, supervised classification, decision-tree rule extraction, clustering, association-rule mining, external-reference comparison, and graph construction.

### Layer-3: Mining and modeling

The study evaluates descriptive degradation signatures, session-grouped supervised degradation classifiers, pruned decision-tree rules, clustering profiles, and association rules.

### Layer-4: Graph-based analysis

Three graph views are constructed: a warehouse measurement graph, a context co-occurrence degradation graph, and an external-reference evidence graph. The graph layer summarizes degradation-related context pairs, organizes external-reference evidence, and evaluates auxiliary transition-graph neural benchmarks over warehouse measurement nodes.

## Data sources

| Dataset | Role |
|---|---|
| SynNetQoS | Main dataset for warehouse construction, degradation labeling, mining, supervised modeling, graph construction, and graph neural benchmarking. |
| Vienna 4G/5G | External reference for selected signal/RSRP and throughput comparisons. |
| Campus QoS | External reference for controlled throughput, jitter, packet-loss, and offered-traffic comparisons. |
| UCC 5G Context | External reference for production-trace signal, mobility, application context, and throughput comparisons. |
| 5G-LENA/ns-3 | Controlled simulator reference for selected service-performance checks. |

## Repository structure

```text
warehoused-network-degradation-mining/
├─ config/                         # Dataset registry and path configuration
├─ data/
│  ├─ raw/                         # Raw datasets and external references
│  ├─ interim/                     # Cleaned intermediate tables
│  └─ processed/                   # Warehouse, mining, and graph tables
├─ figures/                        # Generated figures
├─ results/                        # Audits, model outputs, mining summaries, and graph outputs
├─ scripts/                        # Reproducible pipeline entry points
├─ src/network_degradation_mining/ # Reusable project modules
└─ tests/                          # Repository tests
```

## Pipeline

Run the pipeline from the repository root. Stage 02 prepares the external-reference tables. When regenerating the 5G-LENA/ns-3 reference from a local ns-3 build, set `NS3_ROOT` before running Stage 02.

```bash
python scripts/00_audit_raw_inputs.py
python scripts/01_prepare_synnetqos_core.py

export NS3_ROOT=/path/to/ns-3
python scripts/02_prepare_external_references.py

python scripts/03_build_warehouse_tables.py
python scripts/04_create_mining_feature_tables.py
python scripts/05_descriptive_warehouse_mining.py
python scripts/06_supervised_classification.py
python scripts/07_decision_tree_rule_extraction.py
python scripts/08_clustering_analysis.py
python scripts/09_association_rule_mining.py
python scripts/10_external_reference_comparison.py
python scripts/11_build_graph_tables.py
python scripts/12_graph_mining.py
python scripts/13_gnn.py
```

## Pipeline stages

| Stage | Script | Output focus |
|---:|---|---|
| 00 | `00_audit_raw_inputs.py` | Raw input inventory and readability audit |
| 01 | `01_prepare_synnetqos_core.py` | Cleaned SynNetQoS core table |
| 02 | `02_prepare_external_references.py` | Cleaned external-reference tables |
| 03 | `03_build_warehouse_tables.py` | Fact table and dimension tables |
| 04 | `04_create_mining_feature_tables.py` | Supervised, clustering, association-rule, and external-reference summary tables |
| 05 | `05_descriptive_warehouse_mining.py` | Frequency tables, degradation summaries, crosstabs, interaction matrices, and descriptive figures |
| 06 | `06_supervised_classification.py` | Session-grouped supervised degradation-classification benchmarks |
| 07 | `07_decision_tree_rule_extraction.py` | Pruned decision-tree rules and representative decision paths |
| 08 | `08_clustering_analysis.py` | Clustering validation, profiles, stability checks, and cluster figures |
| 09 | `09_association_rule_mining.py` | Frequent itemsets, association rules, and degradation-rule summaries |
| 10 | `10_external_reference_comparison.py` | Selected-variable comparison against external reference sources |
| 11 | `11_build_graph_tables.py` | Warehouse, context co-occurrence, and external-reference graph tables |
| 12 | `12_graph_mining.py` | Graph summaries, context-pair evidence, and external-reference evidence summaries |
| 13 | `13_gnn.py` | Graph-derived baseline and GraphSAGE/GCN transition benchmarks |

## Warehouse layer

The warehouse layer contains one measurement fact table and nine context dimensions.

```text
fact_network_measurement
dim_time
dim_area
dim_device
dim_network
dim_radio
dim_application
dim_mobility
dim_environment
dim_service_state
```

The warehouse design supports reproducible joins for descriptive mining, supervised modeling, association-rule mining, clustering, external-reference comparison, and graph construction.

## Graph layer

The graph layer is built from three views.

| Graph view | Purpose |
|---|---|
| Warehouse measurement graph | Represents measurement nodes, session membership, operating-context links, and within-session transitions. |
| Context co-occurrence degradation graph | Represents pairs of operating-context values with row count, support, degradation rate, baseline rate, and degradation lift. |
| External-reference evidence graph | Organizes source, metric, context, and comparison evidence from the external reference sources. |

The context co-occurrence graph is used for degradation-pattern evidence. The external-reference graph organizes comparison evidence while preserving dataset-specific measurement scope. GraphSAGE and GCN transition benchmarks are evaluated over warehouse measurement-transition structure.

## Generated outputs

Generated outputs are written under `results/` and `figures/`.

| Directory | Contents |
|---|---|
| `results/audits/` | Raw input, cleaning, label, and column-profile audits |
| `results/warehouse/` | Warehouse integrity and column-coverage checks |
| `results/descriptive_mining/` | Degradation-rate summaries, frequency tables, crosstabs, and interaction tables |
| `results/supervised/` | Model metrics, confusion matrices, ranked model tables, and feature importance |
| `results/decision_tree_rules/` | Tree rules, leaf summaries, and representative rule paths |
| `results/clustering/` | Cluster validation, profiles, stability checks, and sensitivity outputs |
| `results/association_rules/` | Frequent itemsets, association rules, and selected degradation-rule summaries |
| `results/external_reference/` | External-reference preparation, readiness, and selected-variable comparison outputs |
| `results/graph/` | Graph-view summaries, context-pair evidence, external-reference graph summaries, and graph neural benchmark outputs |

## Figures

The repository includes generated figures for the warehouse, mining, modeling, external-reference, and graph-analysis layers.

![Warehouse star schema diagram](image.png)
<p align="center"><em>Warehouse schema used to organize measurement records and operating-context dimensions.</em></p>

![Top degradation context-pair lift](image-1.png)
<p align="center"><em>Highest-lift operating-context pairs associated with degraded service states.</em></p>

![Supervised model comparison](image-2.png)
<p align="center"><em>Session-grouped supervised degradation-classification performance.</em></p>

Additional generated figures are stored under `figures/descriptive_mining/`, `figures/external_reference/`, `figures/graph/`, `figures/supervised/`, and `figures/warehouse/`.

## Installation

Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project in editable mode with development dependencies.

```bash
python -m pip install -e ".[dev]"
```

Install optional graph-neural dependencies when running the transition-GNN benchmark.

```bash
python -m pip install -e ".[gnn]"
```

## Reproducibility checks

Run the test suite.

```bash
python -m pytest tests
```

Run Ruff.

```bash
python -m ruff check scripts src/network_degradation_mining tests
```

Inspect changed files after regenerating outputs.

```bash
git status -sb
git diff --name-status
```

## Data and licensing

Raw datasets are expected under `data/raw/` according to `config/paths.yaml` and `config/dataset_registry.yaml`.

Third-party datasets remain under their original licenses and citation requirements. This repository does not relicense third-party raw data.

Source code in this repository is released under the Apache License 2.0. Documentation, figures, and generated result tables authored in this repository are released under Creative Commons Attribution 4.0 International unless a file states otherwise. Third-party datasets, raw inputs, and external reference sources are governed by their own licenses and terms.