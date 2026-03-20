# TargetPrior
This repository contains the source code and analysis scripts for the publication: "TargetPrior: A miRNA-Signature Embedded Evolutionary Learning Framework for Prioritizing Drug Targets in Acute Myeloid Leukemia".

The framework includes strictly paired multi-omics data preprocessing, evolutionary learning-based stability-aware feature selection, survival analysis, and topology-driven network visualization. TargetPrior systematically navigates the underdetermined solution space using a Three-Layer Architecture to identify stable topological anchors for downstream network prioritization.

Contact: For feedback or feature requests, please contact the author: Shinn-Ying Ho (syho@nycu.edu.tw)

## Citation
If you use TargetPrior in your research, please acknowledge it by citing:

> TargetPrior: A miRNA-Signature Embedded Evolutionary Learning Framework for Prioritizing Drug Targets in Acute Myeloid Leukemia

## Key Capabilities
TargetPrior is designed to overcome the "curse of dimensionality" in multi-omics data. Key features include:
* **Three-Layer Evolutionary Learning (EL-CAML):** Systematically navigates the degenerate solution space to extract a parsimonious signature and complementary biomarkers, effectively mitigating the 'Rashomon effect' inherent to accuracy-driven heuristic methods.
* **Biological De-noising and Topological Anchoring:** Filters out topological noise (the 'hairball effect') by utilizing stability-derived miRNAs as upstream anchors to reconstruct highly specific downstream regulatory networks.
* **Prioritizing 'Undruggable' Vulnerabilities:** Systematically identifies master regulators (e.g., AKT1, TOP2A, transcription factors, and phosphatases) via network centrality, rigorously benchmarked against clinical gold standards using Mean Reciprocal Rank (MRR).
* **Paired Multi-Omics Integration:** Seamlessly integrates patient-matched miRNA, mRNA, and experimentally validated (miRTarBase) drug-target interaction data to capture authentic context-specific regulatory mechanics rather than spurious population-level correlations.

## System Requirements
To reproduce the results presented in the manuscript, the following environments are required:

* **Python**: v3.10 (Core libraries: `scikit-learn` v1.1.2, `numpy`, `pandas`)
* **MATLAB**: Required for signature-DEG mapping and scatter plotting (`.m` scripts).
* **R**: Required for Differential Expression Log2 Fold Change calculations (`.R` scripts).
* **OS**: The framework has been tested on `Windows 10/11`.

## Usage Guide
Please refer to the documentation for details regarding input file formats and guidelines for parallel processing.

## Workflow and Quick Start
 The analysis pipeline generally follows this sequence:

1.  **Paired Data Preprocessing**: 
    Run scripts in `miRNA_dataset/` to prepare normalized matrices.
2.  **Three-Layer Feature Selection (EL-CAML)**: 
    Execute `EL_algorithms/` to identify the optimal stability-driven miRNA signature.
3.  **Out-of-Sample Validation & Survival Analysis**: 
    Use `KaplanMeier_plot/KaplanMeier_plot.py` to validate cross-cohort generalizability and ordinal risk ranking.
4.  **Topology-Driven Network Assembly**: 
    Map the signature to DEGs using `signatureAndDEGs/` and visualize the stability-anchored regulatory landscape with `network/`.


## Directory Structure & File Descriptions
### `miRNA_dataset/`
Contains the raw and processed miRNA datasets sourced from the TARGET-AML and CCG2961 pediatric cohorts.

`miRNA_Dataset.xlsx`: Original miRNA profile (samples with survival time < 30 days excluded).

`miRNA_Dataset_logtransform.xlsx`: Log2-transformed miRNA profile.

`train_pre.csv`: Normalized training set (contains AAML0531 and AAML03P1).

`ind_pre_large.csv`: Normalized internal test set (contains AAML0531 and AAML03P1).

`ind_pre_small.csv`: STD-normalized independent external validation set (contains CCG2961 only).

`ind_pre_total.csv`: Complete normalized test set (merged file of ind_pre_large.csv and ind_pre_small.csv).

### `EL_algorithms/`
Contains the core programs for evolutionary learning (EL) based feature selection and SVM parameter optimization. This includes modules for data pre-processing, genetic algorithms, orthogonal experiments, and Orthogonal Array Crossover, implementing the stability-driven global search strategy (IBCGA).

`normalization.py`: Input method for normalizing training and test sets.

`EL_algorithm.py`: Main program for selecting features and SVM parameters via evolutionary learning.It explicitly computes the selection frequency across 30 independent runs to evaluate algorithmic stability.
Input: Normalized training and test sets.
Output: Calculation process and results stored in EL_output.txt.

`oa.py`: Orthogonal experiment function. Inputs the characteristic number and outputs the relevant orthogonal experiment table.

`oac.py`: Orthogonal Array Crossover module based on EL Crossover, utilizing Orthogonal Array to optimize training.


### `KaplanMeier_plot/`
Contains scripts for Kaplan-Meier survival analysis, utilizing the miRNA signature and EL-AML prediction results.

`KaplanMeier_plot.py`: Script for plotting survival curves. It uses the predicted decision value to classify high vs. low risk across distinct cohorts.

`train_pre_sur.csv`: Normalized training set for survival analysis (AAML0531 and AAML03P1).

`ind_pre_large_sur.csv`: Normalized main test set for survival analysis (AAML0531 and AAML03P1).

`ind_pre_small_sur.csv`: Normalized independent test set (CCG2961).

`ind_pre_total_sur.csv`: Normalized complete test set (merged ind_pre_large.csv and ind_pre_small.csv).


### `method_comparison/`
Contains scripts to benchmark EL-CAML against state-of-the-art machine learning models and feature selection methods to evaluate the generalization gap and out-of-sample robustness.

`method_comparison.py`: Script for running benchmarking predictions. Outputs model evaluation indicators and data for plotting ROC curves.

`boosting/`: Methods and results using the classic Boosting model without feature selection (full-feature architecture).

`lasso/`: Methods and results using cross-validated Lasso feature selection.

`pvalue/`: Methods and results using statistical p-value feature selection (representing local variance metrics).

`EL-CAML/`: Model results established in this study. Includes baseline comparisons against Deep Learning (MLP) and Boruta.


### `signatureAndDEGs/`
Contains scripts for comparing biomarker identification between EL and statistical methods, and statistically verifying target specificity via permutation testing.

`signatureAndDEGs.m`: MATLAB script to calculate the signature and corresponding DEG count.
Input: Signatures from different methods.
Output: Number of DEGs (executed 30 times).

`EL-CAML.xlsx`: Signatures selected by EL-CAML and their corresponding DEG quantities.

`p-value.xlsx`: Signatures selected by statistical methods and their corresponding DEG quantities.

`plot_scatter.m`: Script to plot signatures vs. DEG counts (Input: output.xlsx from signatureAndDEGs.m).


### `miRNA_biomarker_AUC/`
Contains scripts and datasets for distinguishing between Normal samples and AML subtypes.
Labels: 0 = Normal, 1 = AML (Non-relapse), 2 = AML (Relapse).

`TARGET-AML_NormalandAML.xlsx`: Dataset for model building.

`miRNA_biomarker_AUC.py`: Script for predicting Normal vs. AML. Models single features and evaluates performance.

`signature_result.xlsx`: Output of model prediction AUC results.

`nornal_aml_indroc_sig/`: Folder containing cross-validation ROC curves.

`nornal_aml_roc_sig/`: Folder containing test set ROC curves.


### `logFC/`
Contains scripts for calculating Log2 Fold Change for both miRNA and mRNA data types to establish the stringent differential thresholds required for maximizing the topological signal-to-noise ratio.

`miRNA/` & `mRNA/`: Folders containing respective calculation programs and input datasets.

`logfc_work.R`: R script using a package to calculate Log2 Fold Change for non-class features.

`TARGET-AML_miRNA.csv` / `TARGET-AML_mRNA.csv`: Input datasets for analysis.

`miRNA_BHmethod.csv` / `mRNA_BHmethod.csv`: Output files containing calculated Log2 Fold Change results.


### `database/`
Contains annotated gene data and final organized tables.

`annotation_ensembl_geneID.xlsx`: Mapping of Ensembl IDs to Gene IDs.

`annotation_gene_position.xlsx`: Gene IDs and soft chromosome information.

`annotation_miRTarBase_hsa_MTI.xlsx`: miRNA IDs and corresponding Gene IDs.

`annotation_miRTarBase_ensembl_TCGA.xlsx`: miRNA IDs, names, and related Gene IDs (arranged by TCGA mRNA data ID).

`annotation_MP_TP.xlsx`: Annotations for membrane proteins, transcription factors, transcription cofactors, and their families.

`annotation_platinum_resistance.xlsx`: Annotations for platinum resistance and related gene changes.

`total_miRNA_gene_annotation.xlsx`: Integrated annotation data (miRNA, genes, membrane proteins, transcription factors, transcription cofactors, and platinum resistance).


### `network/`
Contains network graphs refined using TargetPrior's stability-anchored signatures.

`network.xlsx`: Integrated data for biomarkers identified by EL-CAML serving as upstream topological anchors (miRNA, genes, membrane proteins, transcription factors, platinum resistance).

`drugbank_humdrug.xlsx`: DrugBank database extract for human-related drugs/compounds, indexed by gene.

`networkDEGs_humanDrug.xlsx`: Network identified by EL-CAML supplemented with DrugBank records.

`miRNA_DEG_drug_network.cys`: Cytoscape network file using network.xlsx as input.
Visualizes miRNA, genes, Log2FC, membrane proteins, transcription factors, and platinum resistance info using centrality metrics (Degree, Closeness, Betweenness) to benchmark gold-standard targets.

Gene correlations are imported via STRING.
