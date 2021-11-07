# Data Science in Malicious Files Search: Project Overview
Created a tool that predicts whether a file is malicious based on a list of statically imported .exe file libraries.
* Preprocessed data for analysis
* Conducted a comparative analysis of different classifiers
* Illustrated quality of classifiers by PR and ROC curves
* Compiled a report on the quality of classifiers based on validation data (validation.txt)
* Created a forecast of the file's malignancy based on the imported libraries (1 — if the file is malicious, 0 - otherwise) (prediction.txt)
* Created an explanation file, where for each line of the test sample file contains the reason why the model considered this file malicious (the lines for non-malicious files remain empty) (explain.txt)

## Code and Resources Used
* Python Version: 3.9.5
* Packages: pandas, numpy, sklearn, matplotlib, pickle

## Data Preprocessing
The samples are presented in the form of .tsv files with three columns:
* **is_virus** – whether the file is malicious: 1=yes, 0=no; 
* **filename** - the file name for review; 
* **libs** - comma-separated enumeration of libraries statically imported by this file (the LIEF library was used to get the list).

I needed to convert data so that it was usable for our models. I made the following changes:
* Convert text data to a matrix of token counts
* Split the data into train subsets

## Exploratory data analysis
I looked at the distributions of the data and class balance.

![](https://user-images.githubusercontent.com/59449626/140651094-fe5c9750-f008-4fdc-9176-3da2cefc2ea0.png)

## Model Building
I had train, validation and test samples, so i didn't need split the data into train and tests sets.

I tried four different classifiers and evaluated them using F-score. 

* **Logistic Regression
* **Decision Tree Classifier
* **K-Neighbors Classifier
* **SGD Classifier

Below are a few highlights from the pivot tables.


