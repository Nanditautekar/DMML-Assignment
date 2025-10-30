import json
from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame
import numpy as np
from utils.logger import logger

class DataValidator:
    def __init__(self, config:dict):
        """
        Data validator class
        :param config: Dictionary with reference values for dtypes, range etc.
        """
        self.data:DataFrame = None
        self.config = config

    @abstractmethod
    def load(self, **kwargs):
        return NotImplemented

    def validate(self):
        reports = {}
        logger.info("Checking for missing values")
        reports["MissingValues"] = self.check_missing_values(self.data)
        logger.info("Checking for duplicates")
        reports["Duplicates"] = self.check_duplicates(self.data)
        logger.info("Validating data types")
        reports["DataTypeValidation"] = self.validate_data_types(self.data, self.config["dtypes"])
        logger.info("Validating data ranges")
        reports["RangeValidation"] = self.validate_ranges(self.data, self.config["ranges"])
        return reports

    @staticmethod
    def generate_data_quality_report(reports:dict, output_file):
        """Generate a comprehensive data quality report.
        :param output_file: Excel file path
        :param reports: Dict of quality reports
        """
        # Missing values check
        missing_report = reports["MissingValues"]

        # Duplicates check
        duplicate_count, duplicates = reports["Duplicates"]
        duplicates_summary = pd.DataFrame([{'DuplicateCount': duplicate_count}])

        # Data types check
        dtype_report = reports["DataTypeValidation"]

        # Range check
        range_report = reports["RangeValidation"]

        # Save each part of the report into separate sheets in an Excel file
        with pd.ExcelWriter(output_file) as writer:
            missing_report.to_excel(writer, sheet_name='MissingValues', index=False)
            duplicates_summary.to_excel(writer, sheet_name='Duplicates', index=False)
            dtype_report.to_excel(writer, sheet_name='DataTypeValidation', index=False)
            range_report.to_excel(writer, sheet_name='RangeValidation', index=False)
        print(f"Data quality report generated: {output_file}")

        # Return the reports as a dictionary in case further processing is needed
        report = {
            'MissingValues': missing_report,
            'Duplicates': duplicates_summary,
            'DataTypeValidation': dtype_report,
            'RangeValidation': range_report
        }
        return report

    @staticmethod
    def check_missing_values(df):
        """Check for missing values in each column."""
        missing = df.isnull().sum()
        missing_report = missing[missing > 0].reset_index()
        missing_report.columns = ['Column', 'MissingCount']
        return missing_report

    @staticmethod
    def check_duplicates(df):
        """Identify duplicate rows in the dataframe."""
        duplicate_count = df.duplicated().sum()
        duplicates = df[df.duplicated()]
        return duplicate_count, duplicates

    @staticmethod
    def validate_data_types(df, expected_dtypes):
        """
        Validate data types based on expected_dtypes which is a dictionary
        where key is column name and value is expected dtype.
        """
        dtype_report = []
        for col, expected in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = df[col].dtype
                status = "OK" if np.issubdtype(actual_dtype, expected) else "Mismatch"
                dtype_report.append({
                    'Column': col,
                    'Expected': expected.__name__ if hasattr(expected, '__name__') else str(expected),
                    'Actual': actual_dtype,
                    'Status': status
                })
        return pd.DataFrame(dtype_report)

    @staticmethod
    def validate_ranges(df, range_checks):
        """
        Validate if the values in certain columns fall within the expected range.
        range_checks is a dictionary with:
            key: column name
            value: tuple (min_value, max_value)
        """
        def gt(val, limit):
            try:
                if float(val) > limit:
                    return True
                return False
            except:
                return False
        def lt(val, limit):
            try:
                if float(val) < limit:
                    return True
                return False
            except:
                return False
        range_report = []
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                below_min = len(list(filter(lambda x: lt(x, min_val), df[col])))
                above_max = len(list(filter(lambda x: gt(x, max_val), df[col])))
                total_issues = below_min + above_max
                range_report.append({
                    'Column': col,
                    'ExpectedMin': min_val,
                    'ExpectedMax': max_val,
                    'BelowMin': below_min,
                    'AboveMax': above_max,
                    'TotalOutOfRange': total_issues
                })
        return pd.DataFrame(range_report)


class CSVDataValidator(DataValidator):

    def load(self, source_path:str):
        try:
            self.data = pd.read_csv(source_path)
            logger.info(f"Data loaded successfully from {source_path}.")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

class JSONDataValidator(DataValidator):
    def load(self, source_path):
        try:
            with open(source_path, 'r') as f:
                json_data = json.load(f)
            self.data = pd.DataFrame(json_data)
            logger.info(f"Data loaded successfully from {source_path}.")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
