\# AI-Based Cyber Threat Detection System



\## Overview

An ML-based network traffic analysis system that classifies known cyber threats and identifies anomalous traffic as potential novel threats.



\## Features

\- Known attack classification using XGBoost

\- Anomaly detection using Isolation Forest

\- Confidence-based hybrid threat detection

\- SHAP-based prediction explanations

\- Threat severity assessment

\- MITRE ATT\&CK mapping

\- Security recommendations

\- Interactive Streamlit dashboard



\## Technologies Used

\- Python

\- XGBoost

\- Scikit-learn

\- Isolation Forest

\- SHAP

\- Pandas and NumPy

\- Streamlit



\## Dataset

Uses a CICIDS2017-derived network traffic dataset. Dataset files are excluded from this repository.



\## Run Locally

1\. Install Python.

2\. Create and activate a virtual environment.

3\. Install dependencies:

&#x20;  `pip install -r requirements.txt`

4\. Ensure the required trained models are available in the `models/` folder.

5\. Run the application:

&#x20;  `streamlit run app/app.py`



\## Project Limitation

This prototype analyzes uploaded Parquet traffic records. It does not currently implement continuous live packet capture.

