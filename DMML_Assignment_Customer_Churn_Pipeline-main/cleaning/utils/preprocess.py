import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cleaning.utils.logger import logger



class DataProcessor:
    def __init__(self, path, config):
        """
        Initialize the DataProcessor with the dataset.

        Parameters:
        filepath (str): Path to the CSV file containing the dataset.
        config (dict): Config for data fields
        """
        self.df = None
        self.cleaned_df = None
        self.preprocessed_df = None
        self.config = config
        self.scaler = None
        self.scaler_mapping = None
        self.__load_data(path)
        self.display_initial_summary()

    def __load_data(self, path):
        # Load data from data folders
        for folder in os.walk(path):
            if 'CSV' in folder[0]:
                source_type = "CSV"
            elif 'JSON' in folder[0]:
                source_type = 'JSON'
            else:
                continue
            files = folder[-1]
            for file in files:
                file_path = os.path.join(folder[0], file).__str__()
                logger.info(f"Loading data from {file_path}")
                if source_type == 'CSV':
                    df = pd.read_csv(file_path)
                else:
                    with open(file_path, 'r') as f:
                        df = pd.DataFrame(json.load(f))
                if isinstance(self.df, pd.DataFrame):
                    self.df = pd.concat([self.df, df])
                else:
                    self.df = df


    def display_initial_summary(self):
        """
        Display initial data information, including data types, missing values, and summary statistics.
        """
        print("Data Info:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nSummary Statistics:")
        print(self.df.describe(include='all'))

    def __remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        This method uses the pandas drop_duplicates function.
        """
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        final_shape = self.df.shape
        logger.info(f"Duplicates removed: {initial_shape[0] - final_shape[0]} duplicate rows dropped.")
        return self.df

    def __remove_irrelevant_columns(self):
        """
        Remove irrelevant fields that do not contribute to churn model training.

        Parameters:
        irrelevant_fields (list): A list of column names to drop. If None, defaults to dropping 'CustomerID'.
        """
        irrelevant_fields = self.config["irrelevant_columns"]

        df = self.df.copy()
        for field in irrelevant_fields:
            if field in df.columns:
                df.drop(field, axis=1, inplace=True)
        self.df = df
        logger.info(f"Irrelevant fields {irrelevant_fields} removed.")
        return df

    def __clean_data(self):
        """
        Clean the data by handling missing values, converting data types, and removing duplicates if configured.
        The method treats numerical, binary, and categorical columns separately based on configuration.
        """
        df = self.df.copy()

        # Convert TotalCharges to numeric if applicable
        if "TotalCharges" in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


        # Get columns from config
        numerical_cols = self.config.get("numerical_columns", [])
        categorical_cols = self.config.get("categorical_columns", [])
        binary_cols = self.config.get("binary_columns", [])

        # Impute numerical columns using the specified strategy
        num_strategy = self.config.get("impute_strategy", {}).get("numerical", "median")
        if num_strategy == 'median':
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        elif num_strategy == 'mean':
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

        # Impute binary columns separately
        binary_strategy = self.config.get("impute_strategy", {}).get("binary", "mode")
        for col in binary_cols:
            if col in df.columns:
                if binary_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif binary_strategy == 'constant':
                    fill_value = self.config.get("binary_fill_value", "No")
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Impute non-binary categorical columns separately
        cat_strategy = self.config.get("impute_strategy", {}).get("categorical", "mode")
        for col in categorical_cols:
            if col in df.columns:
                if cat_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif cat_strategy == 'constant':
                    fill_value = self.config.get("categorical_fill_value", "Unknown")
                    df[col] = df[col].fillna(fill_value)
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        self.cleaned_df = df
        logger.info("Data cleaning completed.")
        return df

    def __preprocess_data(self):
        """
        Preprocess the data by standardizing numerical features and converting binary/categorical columns
        to integer codes using the mappings. This treats both binary and categorical columns similarly.
        """
        if self.cleaned_df is None:
            print("Clean the data first by calling clean_data()")
            return

        df = self.cleaned_df.copy()
        # Remove duplicates if enabled in config (default True)
        remove_dups = self.config.get("remove_duplicates", True)
        if remove_dups:
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            logger.info(f"Duplicates removed in clean_data: {initial_shape[0] - df.shape[0]} duplicate rows dropped.")

        numerical_cols = self.config.get("numerical_columns", [])
        binary_cols = self.config.get("binary_columns", [])
        categorical_cols = self.config.get("categorical_columns", [])

        # Build mappings for binary and categorical columns
        self.__build_categorical_mappings()
        for col in binary_cols + categorical_cols:
            if col in df.columns and col in self.categorical_mappings:
                mapping = self.categorical_mappings[col]["mapping"]
                df[col] = df[col].map(mapping)

        # Standardize numerical features using the specified scaling method
        scaling_method = self.config.get("scaling_method", "StandardScaler")
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()  # fallback

        if numerical_cols:
            scaled_vals = scaler.fit_transform(df[numerical_cols])
            df[numerical_cols] = scaled_vals
            # Store the mapping for each numerical column based on the scaler used.
            if scaling_method == "StandardScaler":
                self.scaler_mapping = {col: {"mean": scaler.mean_[i], "scale": scaler.scale_[i]}
                                       for i, col in enumerate(numerical_cols)}
            elif scaling_method == "MinMaxScaler":
                self.scaler_mapping = {col: {"min": scaler.data_min_[i], "scale": scaler.scale_[i]}
                                       for i, col in enumerate(numerical_cols)}

        self.scaler = scaler  # Save the fitted scaler for inference
        self.preprocessed_df = df
        print("Data preprocessing completed.")
        return df

    def __build_categorical_mappings(self):
        """
        Build a mapping for each categorical column specified in the configuration.
        For each column, unique values are sorted and mapped to integers starting at 0.
        The mapping for each column is stored in self.categorical_mappings.
        """
        mappings = {}
        for col in self.config.get("categorical_columns", []) + self.config.get("binary_columns", []):
            if col in self.cleaned_df.columns:
                # Extract unique values, drop missing values and sort them
                unique_values = sorted(self.cleaned_df[col].dropna().unique().tolist())
                # Create mapping: each unique value gets an integer starting from 0
                col_mapping = {value: idx for idx, value in enumerate(unique_values)}
                mappings[col] = {
                    "mapping": col_mapping,
                    "values": unique_values
                }
                logger.info(f"Mapping for column '{col}': {col_mapping}")
        self.categorical_mappings = mappings
        return mappings

    def preprocess_for_inference(self, new_data):
        """
        Preprocess new (inference) input data so that it is consistent with the training data.
        Uses the saved scaler and categorical mappings.

        Parameters:
        new_data (pd.DataFrame): New input data for inference.

        Returns:
        pd.DataFrame: Preprocessed data ready for inference.
        """
        df = new_data.copy()

        # Remove irrelevant fields
        irrelevant_fields = self.config.get("irrelevant_columns", [])
        for field in irrelevant_fields:
            if field in df.columns:
                df.drop(field, axis=1, inplace=True)

        # Impute missing values for numerical columns using training medians
        numerical_cols = self.config.get("numerical_columns", [])
        for col in numerical_cols:
            if col in df.columns:
                median_val = self.cleaned_df[col].median() if self.cleaned_df is not None else df[col].median()
                df[col] = df[col].fillna(median_val)

        # Impute missing values for binary and categorical columns using training modes
        binary_cols = self.config.get("binary_columns", [])
        categorical_cols = self.config.get("categorical_columns", [])
        for col in binary_cols + categorical_cols:
            if col in df.columns:
                mode_val = self.cleaned_df[col].mode()[0] if self.cleaned_df is not None else df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

        # Apply mappings for binary and categorical columns (map unseen values to -1)
        if not self.categorical_mappings:
            self.__build_categorical_mappings()
        for col in binary_cols + categorical_cols:
            if col in df.columns:
                mapping = self.categorical_mappings.get(col, {}).get("mapping", {})
                df[col] = df[col].apply(lambda x: mapping.get(x, -1))

        # Scale numerical columns using the fitted scaler
        if self.scaler is not None and numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def process(self, config:dict=None):
        if config:
            self.config = config
        self.__remove_duplicates()
        self.__clean_data()
        self.__remove_irrelevant_columns()
        self.__preprocess_data()
        return self.preprocessed_df



    def visualize_histogram(self, column, bins=30):
        """
        Visualize the distribution of a numerical feature with a histogram and KDE plot.

        Parameters:
        column (str): Column name of the numerical feature.
        bins (int): Number of bins for the histogram.
        """
        if column not in self.df.columns:
            logger.info(f"Column {column} not found in the dataset.")
            return

        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column], bins=bins, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def visualize_boxplot(self, column):
        """
        Create a box plot for a numerical feature to identify outliers.

        Parameters:
        column (str): Column name of the numerical feature.
        """
        if column not in self.df.columns:
            logger.info(f"Column {column} not found in the dataset.")
            return

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=self.df[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.show()


