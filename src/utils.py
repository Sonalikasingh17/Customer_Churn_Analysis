import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            test_model_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def create_feature_aggregations(transaction_df):
    '''Create aggregated features from transaction history'''
    try:
        # Convert TransactionDate to datetime
        transaction_df['TransactionDate'] = pd.to_datetime(transaction_df['TransactionDate'])

        # Calculate aggregated features per customer
        agg_features = transaction_df.groupby('CustomerID').agg({
            'AmountSpent': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'TransactionDate': ['min', 'max']
        }).reset_index()

        # Flatten column names
        agg_features.columns = ['CustomerID', 'total_spent', 'avg_spent', 'std_spent', 
                               'transaction_count', 'min_spent', 'max_spent', 
                               'first_transaction', 'last_transaction']

        # Calculate days between first and last transaction
        agg_features['transaction_period_days'] = (
            agg_features['last_transaction'] - agg_features['first_transaction']
        ).dt.days

        # Fill NaN values
        agg_features['std_spent'] = agg_features['std_spent'].fillna(0)
        agg_features['transaction_period_days'] = agg_features['transaction_period_days'].fillna(0)

        # Create category-wise spending features
        category_features = transaction_df.groupby(['CustomerID', 'ProductCategory'])['AmountSpent'].sum().unstack(fill_value=0)
        category_features.columns = [f'spent_{cat.lower()}' for cat in category_features.columns]
        category_features = category_features.reset_index()

        # Merge aggregated features
        final_features = agg_features.merge(category_features, on='CustomerID', how='left')

        # Drop date columns
        final_features = final_features.drop(['first_transaction', 'last_transaction'], axis=1)

        return final_features

    except Exception as e:
        raise CustomException(e, sys)
