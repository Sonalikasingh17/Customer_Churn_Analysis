import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import create_feature_aggregations


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from Excel file
            file_path = "Customer_Churn_Data_Large.xlsx"

            # Read all sheets
            customer_demographics = pd.read_excel(file_path, sheet_name='Customer_Demographics')
            transaction_history = pd.read_excel(file_path, sheet_name='Transaction_History')
            customer_service = pd.read_excel(file_path, sheet_name='Customer_Service')
            online_activity = pd.read_excel(file_path, sheet_name='Online_Activity')
            churn_status = pd.read_excel(file_path, sheet_name='Churn_Status')

            logging.info('Read the dataset as dataframes from Excel sheets')

            # Create transaction aggregated features
            transaction_features = create_feature_aggregations(transaction_history)
            logging.info('Created transaction aggregated features')

            # Create customer service features
            cs_features = customer_service.groupby('CustomerID').agg({
                'InteractionType': 'count',
                'ResolutionStatus': lambda x: (x == 'Resolved').sum()
            }).reset_index()
            cs_features.columns = ['CustomerID', 'total_interactions', 'resolved_interactions']
            cs_features['resolution_rate'] = cs_features['resolved_interactions'] / cs_features['total_interactions']
            cs_features['resolution_rate'] = cs_features['resolution_rate'].fillna(0)

            # Merge all dataframes
            df = customer_demographics.merge(transaction_features, on='CustomerID', how='left')
            df = df.merge(cs_features, on='CustomerID', how='left')
            df = df.merge(online_activity, on='CustomerID', how='left')
            df = df.merge(churn_status, on='CustomerID', how='left')

            # Fill missing values
            df = df.fillna(0)

            logging.info('Merged all datasets successfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ChurnStatus'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Data ingestion completed. Train data: {train_data}, Test data: {test_data}")
