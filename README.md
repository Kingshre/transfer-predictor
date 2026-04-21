# CA Transfer Outcome Predictor

A machine learning project that predicts which student demographic groups 
are at risk of below-average UC/CSU transfer rates across California 
community colleges, surfacing equity gaps using official CCCCO data.

## Features
- Random Forest classifier trained on 847 CCCCO student outcome records 
  achieving 83% accuracy
- Equity analysis of transfer rates across 15 California community colleges 
  by ethnicity, 2015–2022
- Interactive Streamlit dashboard with transfer rate charts by ethnicity 
  and college
- SHAP explainability showing college and ethnicity as top predictors
- Live prediction tool for any college/ethnicity/year combination

## Personal Connection
Built by a De Anza College transfer student heading to UC Berkeley Fall 2026 
— this project is personal.

## Tech Stack
Python · pandas · scikit-learn · Streamlit · Plotly · SHAP

## Data Source
[California Community Colleges Chancellor's Office (CCCCO) Student Success 
Metrics via CalPassPlus](https://www.calpassplus.org/launchboard/student-success-metrics.aspx)
