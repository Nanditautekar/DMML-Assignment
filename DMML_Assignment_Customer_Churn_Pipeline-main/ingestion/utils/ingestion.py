import os
import uuid
import json
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import pandas as pd
import requests
from .logger import logger

class DataIngestion(ABC):
    @abstractmethod
    def ingest(self, **kwargs):
        return NotImplemented

    @staticmethod
    def get_filename_str():
        return f'{uuid.uuid4()}__{datetime.now().strftime("%Y%m%d")}'

class CSVDataIngestion(DataIngestion):
    """
    Ingest data from csv files
    """
    @classmethod
    def ingest(cls, file_path, output_dir):
        try:
            data = pd.read_csv(file_path)
            logger.info("CSV ingestion successful: %d records ingested.", len(data))
            # Create output dir if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save raw data with timestamp
            raw_file = os.path.join(output_dir, f'{cls.get_filename_str()}.csv')
            data.to_csv(raw_file, index=False)
            logging.info("CSV raw data saved to %s", raw_file)
            return data, raw_file
        except Exception as e:
            logger.error("CSV ingestion failed: %s", str(e))
            return None

class APIDataIngestion(DataIngestion):
    """
    Ingest data from REST APIs
    """
    @classmethod
    def ingest(cls, api_url, output_dir, **kwargs):
        try:
            headers = kwargs.get('headers')
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()  # Raises error for bad responses
            data_json = response.json()
            logger.info("API ingestion successful: %d records ingested.", len(data_json))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save raw data with timestamp
            raw_file = os.path.join(output_dir, f'{cls.get_filename_str()}.json')
            with open(raw_file, 'w') as f:
                json.dump(data_json, f)
            logging.info("JSON raw data saved to %s", raw_file)
            return data_json, raw_file
        except Exception as e:
            logger.error("API ingestion failed: %s", str(e))
            return None
