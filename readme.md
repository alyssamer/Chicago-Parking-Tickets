# Chicago Parking Tickets

Exploratory analysis and classification modeling on 54M+ parking ticket records (1999-2018) using PostgreSQL and Python to predict ticket payment outcomes and find patterns in the data.


## Overview
Tickets were classified into three categories: Paid, Dismissed, and Unpaid. SQL was used to explore and aggregate the full dataset, with a ~~5% sample being used for modeling.


## Model Results
There were 3 classes which were imbalanced, as majority of tickets were paid, so a weighted F1 score was used for comparison.
Todo: Rerun with balanced class weights 


Decision Tree F1: 
 0.4728

Random Forest F1:
 0.7106

SVM F1:          
 0.5123

KNN F1:          
 0.6601

Naive Bayes F1:  
 0.5711



## Findings
- Tableau tba


## Files
- main.py -> cleaning queries, EDA, sampling, and modeling code
- ddl_script.sql -> database setup 
- dataset -> tba

