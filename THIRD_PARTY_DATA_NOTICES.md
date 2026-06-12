# Third-party data and software notices

This file documents the licensing and redistribution boundaries for datasets, external reference materials, simulator tools, and generated outputs associated with this repository.

This file does not replace original license files, dataset records, software licenses, or terms of use. Third-party materials remain governed by their original licenses and source terms.

## Repository-authored materials

Source code authored for this repository is licensed under the Apache License, Version 2.0, as stated in the root `LICENSE` file.

Repository-authored documentation, generated result tables, generated figures, diagrams, and reporting artifacts are licensed according to `CONTENT_LICENSE.md`, unless a specific file states otherwise.

## SynNetQoS

SynNetQoS is an author-controlled synthetic dataset source.

The SynNetQoS source code is licensed under the Apache License, Version 2.0.

The generated SynNetQoS synthetic dataset, generated result summaries, figures, tables, and CSV outputs are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), according to the SynNetQoS `LICENSE-DATA` file.

SynNetQoS DOI/source record:

https://doi.org/10.5281/zenodo.20172326

SynNetQoS files redistributed with this repository or its archival package are limited to generated synthetic data and generated outputs covered by the SynNetQoS data license.

## Campus QoS

The 5G Campus Network QoS dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), according to its source Zenodo record.

Source record:

https://zenodo.org/records/13754300

Raw Campus QoS files are third-party materials. They are not relicensed by this repository. Any redistribution of raw Campus QoS files must preserve the original CC BY 4.0 license and attribution requirements.

This repository uses Campus QoS only as an external reference source and does not claim ownership over the raw Campus QoS dataset.

## Vienna 4G/5G

The Vienna 4G/5G drive-test dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0), according to its source Zenodo record.

Source record:

https://zenodo.org/records/18338399

Raw Vienna 4G/5G files are third-party materials. They are not relicensed by this repository. Any redistribution of raw Vienna files must preserve the original CC BY 4.0 license and attribution requirements.

This repository uses Vienna 4G/5G only as an external reference source and does not claim ownership over the raw Vienna dataset.

## UCC Beyond Throughput 5G

The UCC Beyond Throughput 5G dataset is licensed under the GNU General Public License Version 3.0 (GPL-3.0), according to the source repository license file.

Source repository:

https://github.com/uccmisl/5Gdataset

Raw UCC files are third-party GPL-3.0 materials. They are not relicensed by this repository.

This repository does not redistribute raw UCC files in the default release archive. Any separate redistribution of raw UCC files must preserve the GPL-3.0 license text, source attribution, and citation requirements, and must clearly identify the files as GPL-3.0-covered third-party material.

## ns-3

ns-3 is licensed under the GNU General Public License Version 2.0 (GPL-2.0).

The ns-3 source code is not redistributed in this repository or its default archival package.

Simulator summaries generated from controlled ns-3 runs are repository-generated outputs when they do not contain copied ns-3 source code or other third-party source files.

## 5G-LENA

5G-LENA is treated as GPLv2/GPL-2.0-family ns-3 simulation tooling.

The 5G-LENA source code is not redistributed in this repository or its default archival package.

Simulator summaries generated from controlled 5G-LENA/ns-3 runs are repository-generated outputs when they do not contain copied 5G-LENA or ns-3 source code.

## Generated outputs

Generated outputs under `results/`, `figures/`, and `data/processed/` are repository-generated analysis outputs unless a file states otherwise.

These generated outputs are distributed under the repository content license only to the extent permitted by the licenses and terms of the source materials from which they were derived.

Generated outputs do not relicense the underlying raw third-party datasets, simulator tools, or copied third-party files.

## Release and archival boundary

The default repository release and archival package include repository source code, configuration files, tests, documentation, generated result tables, generated figures, SynNetQoS files covered by the SynNetQoS data license, processed study outputs, and generated simulator summaries.

The default repository release and archival package do not include:

- raw UCC Beyond Throughput 5G files;
- ns-3 source code;
- 5G-LENA source code;
- third-party simulator or tool source trees;
- low-level simulator traces;
- local exploratory notebooks;
- virtual environments, caches, or system files.

Raw Campus QoS and raw Vienna 4G/5G files are third-party CC BY 4.0 materials. This repository cites their original records and does not treat those raw datasets as repository-authored materials.

## No relicensing of third-party materials

Third-party raw datasets, external reference datasets, simulator tools, and copied third-party files remain governed by their original licenses, access terms, and citation requirements.

Nothing in this repository’s `LICENSE`, `CONTENT_LICENSE.md`, or generated outputs relicenses third-party raw data or third-party software.