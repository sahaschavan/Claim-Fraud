# Auto Insurance Claims: Fraud Analysis
I took a real auto insurance claims dataset from Kaggle and used SQL to answer business questions, used pandas/seaborn to explore the data, and scikit-learn to try to predict which claims get flagged as fraud.
Data: *Auto Insurance Claims Data* from Kaggle (insurance_claims.csv, 1,000 claims, 39 columns: policy details, the insured person, the incident, claim amounts, and a fraud_reported Y/N label).
Tools: Python, pandas, SQLite (SQL), scikit-learn, matplotlib, seaborn

## Breakdown
**1. Cleaning (`load_data.py`)** - I dropped an empty trailing column, fixed column names, turned the `?` placeholders into an "Unknown" category instead of throwing the rows away, made the dates SQL-friendly, and created a 0/1 `fraud` target.

**2. SQL (`sql/queries.sql`)** - 10 queries: fraud rate by incident type/severity/age band/hobby, claims above their type's average, top 3 claims per state, weekly trend with `LAG()`,
and a simple needs review list. I have saved all the results as csv in `outputs/sql_results/`.

**3. EDA (`eda_charts.py`)** - I have used matplot to create 4 charts: fraud rate by severity, fraud rate by hobby, claim amounts for fraud vs non-fraud and correlation heatmap.

**4. ML (`ml_models.py`)** - I have used 6 Ml models as following:
1. Logistic Regression: It gives every feature a weight, adds them up, and turns the total into a fraud probability. Accuracy:82.7%, ROC-AUC: 0.862
2. Decision Tree: It asks a chain of yes/no questions until it lands on a fraud or not-fraud answer. Accuracy:83.8% ROC-AUC: 0.782
3. KNN: It finds the 15 most similar past claims and goes with the majority, if most of them were fraud, it predicts fraud. Accuracy:74.8% ROC-AUC: 0.640
4. SVM: It draws a line between fraud and non-fraud claim, bending it where needed, and predicts based on which side a claim falls. Accuracy:81.1% ROC-AUC: 0.833
5. Random Forest: It builds hundreds of decision trees on random slices of the data and takes a vote. Accuracy:82.3% ROC-AUC: 0.838
6. K-means: Rather than being used to predict fraud it groups claims by similarity in age, tenure, premium and claim amount.


