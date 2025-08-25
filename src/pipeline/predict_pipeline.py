import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            pred_proba = model.predict_proba(data_scaled)[:, 1]
            return preds, pred_proba
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: int,
                 Gender: str,
                 MaritalStatus: str,
                 IncomeLevel: str,
                 total_spent: float,
                 avg_spent: float,
                 std_spent: float,
                 transaction_count: int,
                 min_spent: float,
                 max_spent: float,
                 transaction_period_days: int,
                 spent_books: float,
                 spent_clothing: float,
                 spent_electronics: float,
                 spent_furniture: float,
                 spent_groceries: float,
                 total_interactions: int,
                 resolved_interactions: int,
                 resolution_rate: float,
                 LoginFrequency: int,
                 ServiceUsage: str):

        self.Age = Age
        self.Gender = Gender
        self.MaritalStatus = MaritalStatus
        self.IncomeLevel = IncomeLevel
        self.total_spent = total_spent
        self.avg_spent = avg_spent
        self.std_spent = std_spent
        self.transaction_count = transaction_count
        self.min_spent = min_spent
        self.max_spent = max_spent
        self.transaction_period_days = transaction_period_days
        self.spent_books = spent_books
        self.spent_clothing = spent_clothing
        self.spent_electronics = spent_electronics
        self.spent_furniture = spent_furniture
        self.spent_groceries = spent_groceries
        self.total_interactions = total_interactions
        self.resolved_interactions = resolved_interactions
        self.resolution_rate = resolution_rate
        self.LoginFrequency = LoginFrequency
        self.ServiceUsage = ServiceUsage

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "MaritalStatus": [self.MaritalStatus],
                "IncomeLevel": [self.IncomeLevel],
                "total_spent": [self.total_spent],
                "avg_spent": [self.avg_spent],
                "std_spent": [self.std_spent],
                "transaction_count": [self.transaction_count],
                "min_spent": [self.min_spent],
                "max_spent": [self.max_spent],
                "transaction_period_days": [self.transaction_period_days],
                "spent_books": [self.spent_books],
                "spent_clothing": [self.spent_clothing],
                "spent_electronics": [self.spent_electronics],
                "spent_furniture": [self.spent_furniture],
                "spent_groceries": [self.spent_groceries],
                "total_interactions": [self.total_interactions],
                "resolved_interactions": [self.resolved_interactions],
                "resolution_rate": [self.resolution_rate],
                "LoginFrequency": [self.LoginFrequency],
                "ServiceUsage": [self.ServiceUsage]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
