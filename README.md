# RA
for RA object
Code Description for GitHub Upload
This repository contains the core code for multi-omics (metabolomics and transcriptomics) data analysis and predictive modeling, which is divided into three key modules as follows:
1. Transformer-Based Feature Importance Scoring Codes
Files: transformer_metabolomics.py & transformer_transcriptomics.pyThese scripts take the full expression matrix of metabolomics and transcriptomics sequencing data as input, respectively. Built on the Transformer deep learning model, they are designed to calculate and output the importance score for every single feature in the omics datasets, enabling unbiased evaluation of feature contributions without prior screening.
2. Traditional Machine Learning Model Feature Screening Codes
Files: 5 model_metabolomics.py & 5 model_transcriptomics.pyBased on the overlapping differential features identified by Transformer and differential expression analysis, these codes apply five classical machine learning models to perform secondary feature selection. Each model outputs the top 10 most important features, and consistent critical features across all five models are retained for subsequent analysis. These selected features are then intersected with KEGG enrichment-derived key features to obtain the final core differential features.
3. Predictive Modeling and ROC Curve Validation Codes
Files: Revised_ROC_Metabolomics.py & ROC_Transcriptomics.pyUsing the filtered core differential features as input, these scripts construct predictive models for metabolomics and transcriptomics, respectively. They generate standardized ROC curves and corresponding evaluation metrics to validate the classification performance and reliability of the core features, verifying the validity of the multi-omics feature screening strategy and modeling framework.
