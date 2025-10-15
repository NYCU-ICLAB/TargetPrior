# ELCDT-AML
This is the code for the analyses performed in the publication "".
The framework contains 
Please contact the author, Shinn-Ying Ho syho@nycu.edu.tw, if you would like to feedback or feature requests.

## Citation
If you use ELCDT-AML, please acknowledge by citing "".

## A guide to using ELCDT-AML
see documentation about input file formats and guide to use parallel processing.

### The miRNA_dataset folder contains the miRNA dataset. 
* miRNA_Dataset.xlsx is the original miRNA profile with survival time less than 30 days deleted. 
* miRNA_Dataset_logtransform.xlsx is the miRNA profile that performs log2 transformation.
* train_pre.csv is the normalized training set, which contains AAML0531 and AAML03P1. 
* ind_pre_large.csv is the main test set after normalization transformation, which contains AAML0531 and AAML03P1. 
* ind_pre_small.csv is an independent test set that has been STD normalized and converted, which only contains CCG2961. 
* ind_pre_total.csv is the entire test set that has been normalized and converted, and is the merged file of ind_pre_large.csv and ind_pre_small.csv.

### The EL_algorithms folder contains a simple program for selecting features and SVM parameters through evolutionary learning, including data pre-processing module, genetic algorithm module, orthogonal experiment function, and Orthogonal Array Crossover module.
* normalization.py is the input method to be normalized, training set and test set.
* EL_svm.py is a program for selecting features and SVM parameters based on evolutionary learning. The input is the normalized training set and test set. The calculation process and results will be stored and output as a "EL_output.txt" file.
* oa.py is an orthogonal experiment function. The program inputs the characteristic number and outputs the relevant orthogonal experiment table.
* oac.py is an Orthogonal Array Crossover module program based on evolutionary learning Crossover and uses Orthogonal Array to optimize training.


### The KaplanMeier_plot folder contains the analysis of Kaplan Meier program, miRNA signature, and EL-AML prediction results. 
* KaplanMeier_plot.py is a program for analyzing Kaplan Meier. The program uses the predicted decision value as the value of high and low risk to draw the survival curve.
* train_pre_sur.csv is the normalized training set, which contains AAML0531 and AAML03P1.
* ind_pre_large_sur.csv is the main test set after regularization transformation, which contains AAML0531 and AAML03P1.
* ind_pre_small_sur.csv is an independent test set that has been normalized and transformed, which only contains CCG2961.
* ind_pre_total_sur.csv is the entire test set after normalization transformation, which is the merged file of ind_pre_large.csv and ind_pre_small.csv.


### The method_comparison folder contains programs to run other machine learning predictions, normalized miRNA expression levels, and EL-AML prediction results. This test result was analyzed using the entire test set.
* method_comparison.py is a program for running other machine learning predictions. The program outputs the evaluation indicators of the model and outputs the data for drawing the ROC curve.
* The boosting folder contains the methods and results of using the classic boosting machine learning model without feature selection.
* The lasso folder contains methods and results of using lasso feature selection with other classic machine learning models.
* The pvalue folder contains methods and results of using statistical pvalue features to select and match other classic machine learning models.
* EL-CAML is the model result established in this study.


### The signatureAndDEGs folder contains programs for running Comparison of biomarker identification using EL and statistical methods and selecting DEGs corresponding to signatures using different methods.
* signatureAndDEGs.m is a program that calculates the signature and the corresponding DEG number. The program input is the signature of different methods, and outputs the number of DEG corresponding to different methods. The program is executed 30 times.
* EL-CAML.xlsx is the signature selected by EL-CAML and its corresponding DEG quantity.
* p-value.xlsx is the signature selected by the statistical method and its corresponding number of DEGs.
* plot_scatter.m is a program that plots the signature and the corresponding number of DEGs. The program input is the output.xlsx output by signatureAndDEGs.m.


### The miRNA_biomarker_AUC folder contains the program to run the prediction of normal and AML, and their input datasets. Label contains three categories, 0 is a normal sample, 1 is AML without recurrence, and 2 is AML with recurrence.
* TARGET-AML_NormalandAML.xlsx is the data set for input model building. The Label contains three categories: 0 for normal samples, 1 for AML without recurrence, and 2 for AML with recurrence.
* miRNA_biomarker_AUC.py is a program for predicting normal and AML. The program models and predicts a single feature and evaluates its model performance.
* signature_result.xlsx is the output model prediction AUC result.
* The nornal_aml_indroc_sig folder outputs the cross-validation ROC curve.
* The nornal_aml_roc_sig folder outputs the ROC curve of the test set.


### The logFC folder contains a program for calculating Log2 fold Change. The logFC folder contains two data types: miRNA and mRNA.
* The miRNA and mRNA folders contain the program for calculating Log2 fold Change and the required input data sets respectively.
* logfc_work.R is a program for calculating Log2 Fold Change. The program uses a package to calculate Log2 fold Change for non-class features.
* TARGET-AML_miRNA.csv and TARGET-AML_mRNA.csv are the data sets for input analysis.
* miRNA_BHmethod.csv and mRNA_BHmethod.csv are the calculated output Log2 fold Change results.


### The database folder contains the data of annotated genes and the final organized tables.
* annotation_ensembl_geneID.xlsx contains the ensembl ID and gene ID.
* annotation_gene_position.xlsx contains the gene ID and the information of the soft chromosome.
* annotation_miRTarBase_hsa_MTI.xlsx contains miRNA ID and its corresponding gene ID.
* annotation_miRTarBase_ensembl_TCGA.xlsx contains miRNA ID, miRNA name and related gene ID, where genes are arranged as TCGA mRNA data ID.
* annotation_MP_TP.xlsx contains the genes of annotated membrane proteins, transcription factors, transcription cofactors and their families.
* annotation_platinum_resistance.xlsx contains annotations for platinum resistance and its gene-related changes.
* total_miRNA_gene_annotation.xlsx integrates the data of miRNA, genes, membrane proteins, transcription factors, and transcription cofactor platinum resistance.


### The network folder contains the network graph that has been narrowed down using machine learning signatures.
* network.xlsx integrates the data of miRNA, genes, membrane proteins, transcription factors, and transcription cofactor platinum resistance for biomarkers identified by EL-CAML.
* drugbank_humdrug.xlsx is a database of drug-related genes in drugbank. This database extracts drugs and compounds related to humans and arranges them using genes as indexes.
* networkDEGs_humanDrug.xlsx adds drugbank-related drug records using genes as indexes through the network identified by EL-CAML.
* miRNA_DEG_drug_network.cys is a network diagram that uses network.xlsx as input. It integrates miRNA, genes, and annotates log2 fold change, membrane protein, transcription factor, transcription cofactor platinum information, and is presented in a visual way. The correlation between genes is imported and merged into the network graph through STRING.