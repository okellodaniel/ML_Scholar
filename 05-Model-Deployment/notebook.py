import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import logging

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

logging.info("Script started")

# Load data
logging.info("Loading data from CSV")
df = pd.read_csv('data_week_3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Preprocess data
logging.info("Preprocessing data")
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in categorical_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)
df.churn = (df.churn == 'yes').astype(int)

# Split data
logging.info("Splitting data into training and test sets")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Define feature columns
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
    'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
    'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
    'contract', 'paperlessbilling', 'paymentmethod'
]

# Training function


def train(df_train, y_train, C=1.0):
    logging.info("Training model")
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    logging.info("Model trained successfully")

    return dv, model

# Prediction function


def predict(df, dv, model):
    logging.info("Predicting probabilities for data")
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Cross-validation
logging.info("Starting cross-validation")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    logging.info(f'Fold AUC: {auc:.3f}')

logging.info(f'C={C} Mean AUC: {np.mean(scores):.3f} +- {np.std(scores):.3f}')

# Train final model
logging.info("Training final model on full training set")
dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_pred = predict(df_test, dv, model)

# Evaluate final model
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
logging.info(f'Final model AUC: {auc:.3f}')

# Save the model
logging.info(f"Saving model to {output_file}")
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
logging.info("Model saved successfully")

# Load the model
logging.info(f"Loading model from {output_file}")
with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Test model with a random customer
logging.info("Testing model with a random customer")
customer = df_full_train.iloc[random.randint(
    0, len(df_full_train) - 1)].to_dict()
logging.info(f"Found customer data {customer}")
X_customer = dv.transform([customer])

churn_prob = model.predict_proba(X_customer)[0, 1]
logging.info(f'Customer churn probability: {churn_prob:.3f}')
print(f'Customer churn probability: {churn_prob:.3f}')
