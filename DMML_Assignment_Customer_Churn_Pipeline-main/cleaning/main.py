import os
import json
from cleaning.utils.preprocess import DataProcessor

# source_path = "../Dataset/Customer Churn Data"
# output_path = "../Dataset/Processed Data"

# Preprocessing
# Example configuration dictionary for data preprocessing
conf = {
    # Columns that are not relevant for model training (e.g., unique IDs)
    "irrelevant_columns": ["customerID"],

    # Columns that contain binary values (typically represented as 'Yes'/'No')
    "binary_columns": ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn", "MultipleLines" ],

    # Columns that are considered categorical and will be one-hot encoded (after handling binary columns)
    "categorical_columns": [
        "InternetService", "OnlineSecurity","gender",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ],

    # Columns that are numerical and may need scaling or normalization
    "numerical_columns": ["tenure", "MonthlyCharges", "TotalCharges"],

    # Imputation strategies for missing values
    "impute_strategy": {
        "numerical": "median",  # could be 'median' or 'mean'
        "categorical": "mode"  # typically use mode for categorical columns
    },

    # Scaling method for numerical data: e.g., StandardScaler or MinMaxScaler
    "scaling_method": "StandardScaler"
}

# Initialize the DataProcessor with your dataset filepath

def process(source_path, output_path, config):
    processor = DataProcessor(source_path, config)
    processor.process()
    # Optionally, save the preprocessed data
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if processor.preprocessed_df is not None:
        processor.preprocessed_df.to_csv(os.path.join(output_path, 'processed_data.csv'), index=False)
    if processor.cleaned_df is not None:
        processor.cleaned_df.to_csv(os.path.join(output_path, 'data.csv'), index=False)
        with open(os.path.join(output_path, "config.json"), 'w') as f:
            f.write(json.dumps(processor.config))
        with open(os.path.join(output_path, "scale_mapping.json"), 'w') as f:
            f.write(json.dumps(processor.scaler_mapping))
