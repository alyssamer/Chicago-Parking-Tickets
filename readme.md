# Chicago Parking Tickets

Exploratory analysis and classification modeling on 54M+ parking ticket records (1996-2018) using PostgreSQL and Python to predict ticket payment outcomes and find patterns in the data.


## Overview
Tickets were classified into three categories: Paid, Dismissed, and Unpaid. SQL was used to explore and aggregate the full data set, with a ~7% sample of recent data being used for modeling.


## Model Results
The 3 classes were imbalanced, as majority of tickets were paid, so a weighted F1 score was used for comparison.



Random Forest F1: 0.7160

SVM F1: 0.5740

KNN F1: 0.6660

Naive Bayes F1: 0.6008



## Findings
- Payment rates varied significantly by community area
- Notice level and zipcode were the strongest predictors of payment status (RF feature importance)
- Additional fines did not significantly increase payment rates for unpaid tickets
- Random Forest achieved the best F1 score of 0.71, but plateaued there even with additional data and accounting for class imbalance, 
  suggesting strong predictors outside of the data set effected payment status
- Tableau -> tba


## Files
- main.py -> cleaning queries, EDA, sampling, and modeling code
- ddl_script.sql -> database setup 
- plots/ -> confusion matrices for models and feature importance plot
- dataset -> [Parking Tickets Data](https://www.propublica.org/nerds/download-chicago-parking-ticket-data)


## Limitations/Future Work
- License plate hashing allows for finding repeat offenders, however there were many outliers with
  hundreds of tickets that needs cleaning (due to missing plate placeholders and other unknowns)
- Sample was limited to 7% due to local storage constraints, and was from 2000-2018 to get more of the most recent data
