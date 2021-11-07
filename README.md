# Data Science in Malicious Files Search: Project Overview
Created a tool that predicts whether a file is malicious based on a list of statically imported .exe file libraries.
* Preprocessed data for analysis
* Conducted a comparative analysis of different classifiers: Logistic Regression, Decision Tree Classifier, K-Neighbors Classifier, SGD Classifier
* Illustrated quality of classifiers by PR and ROC curves
* Compiled a report on the quality of classifiers based on validation data (validation.txt)
* Created a forecast of the file's malignancy based on the imported libraries (1 — if the file is malicious, 0 - otherwise) (prediction.txt)
* Created an explanation file, where for each line of the test sample file contains the reason why the model considered this file malicious (the lines for non-malicious files remain empty) (explain.txt)

## Code and Resources Used
* Python Version: 3.9.5
* Packages: pandas, numpy, sklearn, matplotlib, pickle

## Data Preprocessing
The samples are presented in the form of .tsv files with three columns:
**is_virus** – whether the file is malicious: 1=yes, 0=no; 
**filename** - the file name for review; 
**libs** - comma-separated enumeration of libraries statically imported by this file (the LIEF library was used to get the list).
