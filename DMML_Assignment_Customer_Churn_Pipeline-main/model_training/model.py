import json
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import joblib

class Model:
    @staticmethod
    def load_data(data_path:str, label_column:str, drop_columns:list):

        data = pd.read_csv(data_path)
        print("Data preview:")
        print(data.head())
        X = data.drop(columns=drop_columns)
        y = data[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train(X_train, X_test, y_train, y_test):

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(y_test, y_pred)
        return model, {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "report":report}

    @staticmethod
    def save_model(model_dir:str, artifacts_dir, model, report):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, 'logistic_regression_model.pkl')
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

        # Save the performance report as a text file.
        report_filename = os.path.join(artifacts_dir, 'performance_report.json')
        with open(report_filename, "w") as file:
            json.dump(report, file)
        print(f"Performance report saved to {report_filename}")



