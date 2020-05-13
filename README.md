# Loan Payoff Prediction using Deep Learning Neural Networks

A deep learning model that predicts whether a borrower will fully repay a loan or
default (charge off), built on the Lending Club loan dataset.

> **Note on dates:** This project was originally completed in May 2020 as part of my
> MS in Business Analytics at California State University, East Bay. It was uploaded
> to GitHub in June 2026 after my previous GitHub account was deleted. Commit dates
> are set to reflect the original completion date.

## Overview

The banking and lending sector is one of the largest parts of the financial industry.
A large share of the population lives on credit, and accurately predicting loan
repayment is critical for lenders to manage credit risk. This project builds a
neural network classifier that predicts loan outcomes from borrower and loan
attributes.

The target variable indicates whether a borrower **fully paid** the loan or had it
**charged off** (the creditor declared the debt unlikely to be collected).

## Dataset

- **Source:** Lending Club loan data on Kaggle
  - https://www.kaggle.com/wordsforthewise/lending-club
- **Size:** ~396,000 observations, 28 attributes
- **Target:** `loan_status` (Fully Paid vs Charged Off)

> The full dataset (`lending_club_loan_two.csv`, ~96 MB) is **not** committed here
> because it exceeds GitHub file size limits and belongs to the original Kaggle
> authors. Download it from the link above and place it in this folder to run the
> notebook. A small data dictionary, `lending_club_info.csv`, is included.

## Steps Performed

1. **Exploratory data analysis** of loan attributes and the target distribution
2. **Handling missing data** across borrower and loan fields
3. **Encoding categorical variables** and creating dummy variables
4. **Data pre-processing** (scaling and train/test split)
5. **Neural network model creation** with Keras / TensorFlow
6. **Model performance evaluation** and conclusions

## Result

The model achieves approximately **89% accuracy** in predicting whether a borrower
will repay the loan.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow / Keras,
Jupyter Notebook.

## Files

- `Loan payoff prediction deep learning neural network model.ipynb` - main notebook
- `*.pdf` - rendered notebook and project report
- `lending_club_info.csv` - feature data dictionary

## Author

Kedar Patil - MS Business Analytics, California State University, East Bay
