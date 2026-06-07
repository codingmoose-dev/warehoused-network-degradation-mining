# Warehouse-Guided Mining and Modeling of Network Service Degradation Patterns

This repository contains the code, configuration, tests, and generated study outputs for a warehouse-guided network service-degradation mining workflow.

The study builds a warehouse-style analytical representation from session-level network-performance records, derives task-specific mining tables, evaluates classical mining and supervised models, compares selected variables against external references, and builds graph views for graph mining and graph neural benchmarking.

The current working title is:

> Warehouse-Guided Mining and Modeling of Network Service Degradation Patterns

Use the longer title, *Warehouse-Guided Mining and Graph-Based Modeling of Network Service Degradation Patterns*, only if the graph and GNN sections become a major part of the manuscript.

## Study scope

The repository is organized around one main analytical dataset and four external reference sources.

| Dataset | Role in this repository |
|---|---|
| SynNetQoS | Main dataset for warehouse construction, degradation labels, descriptive mining, supervised classification, decision-tree rules, clustering, association rules, graph construction, and graph neural benchmarks. |
| Vienna 4G/5G | External reference for selected signal/RSRP and throughput checks. |
| Campus QoS | External reference for controlled throughput, jitter, loss, and offered-traffic checks. |
| UCC 5G Context | External reference for production-trace signal, mobility, application context, and throughput checks. Throughput comparisons use Download-context rows where appropriate. |
| 5G-LENA/ns-3 | Controlled simulator reference for selected service-performance checks. |

External references are not merged into the supervised training table. They are used for selected-variable reference checks and for the external-reference evidence graph.

## Pipeline

Run scripts from the repository root.

```bash
python scripts/00_audit_raw_inputs.py
python scripts/01_prepare_synnetqos_core.py
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

### Stage summary

| Stage | Script | Purpose |
|---:|---|---|
| 00 | `00_audit_raw_inputs.py` | Inventory registered raw inputs and record readable files, rows, columns, and read failures. |
| 01 | `01_prepare_synnetqos_core.py` | Prepare the cleaned SynNetQoS core table. |
| 02 | `02_prepare_external_references.py` | Prepare cleaned Vienna, Campus QoS, UCC 5G Context, and 5G-LENA/ns-3 reference tables. |
| 03 | `03_build_warehouse_tables.py` | Build the warehouse fact table and dimensions. |
| 04 | `04_create_mining_feature_tables.py` | Build supervised, clustering, association-rule, and external-reference summary tables from warehouse outputs. |
| 05 | `05_descriptive_warehouse_mining.py` | Produce frequency tables, degradation-rate tables, cross-tabs, interaction matrices, and descriptive figures. |
| 06 | `06_supervised_classification.py` | Run session-grouped supervised degradation classification with leakage-aware feature exclusions. |
| 07 | `07_decision_tree_rule_extraction.py` | Train pruned decision-tree variants and export rule tables and representative paths. |
| 08 | `08_clustering_analysis.py` | Run KMeans, hierarchical clustering, Gaussian mixture, DBSCAN, OPTICS, stability checks, profile summaries, and cluster figures. |
| 09 | `09_association_rule_mining.py` | Mine Apriori and FP-Growth rules and export selected degradation-rule summaries. |
| 10 | `10_external_reference_comparison.py` | Compare selected metrics against cleaned external references and export tables and figures. |
| 11 | `11_build_graph_tables.py` | Build warehouse measurement, context co-occurrence, and external-reference graph tables. |
| 12 | `12_graph_mining.py` | Summarize graph views, node/edge types, context-pair evidence, and external-reference graph evidence. |
| 13 | `13_gnn.py` | Run graph-derived baseline and GraphSAGE/GCN transition benchmarks over warehouse measurement nodes. |

## Main outputs

The pipeline writes tables and figures under:

```text
results/
figures/
data/interim/
data/processed/
```

Important result groups include:

- `results/audits/` — raw input, cleaning, and label audits.
- `results/warehouse/` — warehouse integrity and column-coverage checks.
- `results/descriptive_mining/` — warehouse mining summaries and interaction tables.
- `results/supervised/` — model metrics, confusion matrices, feature importance, and ranked model tables.
- `results/decision_tree_rules/` — tree rules, leaf summaries, and rule reports.
- `results/clustering/` — cluster validation, profiles, stability, and sensitivity outputs.
- `results/association_rules/` — itemsets, rules, selected degradation rules, and rule-quality summaries.
- `results/external_reference/` — selected-variable reference checks and external-reference audits.
- `results/graph/` — graph summaries, context-pair evidence, external-reference graph summaries, and GNN benchmark outputs.

## Warehouse layer

The warehouse layer uses a fact table and context dimensions:

- `fact_network_measurement`
- `dim_time`
- `dim_area`
- `dim_device`
- `dim_network`
- `dim_radio`
- `dim_application`
- `dim_mobility`
- `dim_environment`
- `dim_service_state`

This design keeps the measurement record separate from time, network, radio, endpoint, application, mobility, environmental, and service-state context.

## Modeling and mining layers

The repository currently includes:

- descriptive warehouse mining;
- supervised degradation classification;
- decision-tree rule extraction;
- clustering analysis;
- association-rule mining;
- selected-variable external-reference comparison;
- warehouse/context/external-reference graph construction;
- graph mining;
- graph neural benchmarking.

The supervised and graph-neural stages use session-grouped splitting where applicable. Target-definition fields are excluded from model features where they would directly restate the degradation label.

## Scope boundaries

The outputs should be reported as dataset-level mining and modeling evidence from the prepared warehouse tables.

Do not describe the results as:

- deployment-ready prediction;
- field validation of real-world 5G behavior;
- evidence that association rules prove network mechanisms;
- evidence that clusters are natural degradation classes;
- evidence that external-reference rows were merged into supervised training;
- evidence that external-reference graph nodes were used for prediction.

The graph neural models are benchmark models over warehouse-derived measurement-transition structure. They are not the central claim unless their results clearly justify that role in the manuscript.

## Installation

Create and activate a virtual environment, then install the project in editable mode.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For graph neural benchmarking, install the optional GNN dependency group if it is not already available in your environment.

```bash
python -m pip install -e ".[gnn]"
```

## Checks

Run the lightweight repository tests:

```bash
python -m pytest tests
```

Run Ruff:

```bash
python -m ruff check scripts src/network_degradation_mining tests
```

After rerunning the full pipeline, check whether tracked outputs changed:

```bash
git status -sb
git diff --name-status
```

Regenerated PDF figures may change only because of PDF metadata. If the figure content is unchanged, restore the figure files before committing:

```bash
git restore -- figures/
```

If CSV or TXT outputs change, inspect the diff before restoring.

## Data and external references

Raw datasets are expected under `data/raw/` according to `config/paths.yaml` and `config/dataset_registry.yaml`.

Third-party datasets remain under their original licenses and citation requirements. This repository does not relicense third-party raw data. When preparing the manuscript, cite the dataset records and source publications for SynNetQoS, Vienna 4G/5G, Campus QoS, UCC 5G Context, and 5G-LENA/ns-3 as applicable.

## License

Source code in this repository is released under the Apache License 2.0. Documentation, figures, and generated result tables authored in this repository are released under Creative Commons Attribution 4.0 International unless a file states otherwise.

Third-party datasets, raw inputs, and external reference sources are governed by their own licenses and terms.
