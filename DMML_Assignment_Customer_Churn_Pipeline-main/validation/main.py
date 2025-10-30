import os
import numpy as np
from utils.validate import JSONDataValidator, CSVDataValidator
source_path = "../Dataset/Customer Churn Data"
report_path = "../reports/Customer Churn Data"

config = {
    "dtypes": {
        'customerID': object,
        'gender': object,
        'SeniorCitizen': np.integer,
        'Partner': object,
        'Dependents': object,
        'tenure': np.integer,
        'PhoneService': object,
        'MultipleLines': object,
        'InternetService': object,
        'OnlineSecurity': object,
        'OnlineBackup': object,
        'DeviceProtection': object,
        'TechSupport': object,
        'StreamingTV': object,
        'StreamingMovies': object,
        'Contract': object,
        'PaperlessBilling': object,
        'PaymentMethod': object,
        'MonthlyCharges': np.floating,
        'TotalCharges': np.floating,  # Converted to numeric above
        'Churn': object
    },
    "ranges": {
        'tenure': (0, 100),  # Tenure should be within 0 and 100 months
        'MonthlyCharges': (0, 1000),  # Monthly charges are expected to be within a reasonable range
        'TotalCharges': (0, 1e6),
        'SeniorCitizen': (0, 1)  # SeniorCitizen should be 0 or 1
    }}

def validate(config, source_path, report_path):
    csv_validator = CSVDataValidator(config=config)
    json_validator = JSONDataValidator(config=config)
    for folder in os.walk(source_path):
        if 'CSV' in folder[0]:
            source_type = "CSV"
        elif 'JSON' in folder[0]:
            source_type = 'JSON'
        else:
            continue

        # Create folders
        path = os.path.join(report_path, '/'.join(folder[0].split("/")[-2:]))
        if not os.path.exists(path):
            os.makedirs(path)

        files = folder[-1]
        for file in files:
            if source_type == 'CSV':
                validator = csv_validator
            else:
                validator = json_validator
            validator.load(os.path.join(folder[0], file))
            reports = validator.validate()
            validator.generate_data_quality_report(reports, os.path.join(path, file.split('.')[0] + '.xlsx'))


