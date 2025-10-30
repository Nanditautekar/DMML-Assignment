
from ingestion.utils.ingestion import APIDataIngestion, CSVDataIngestion
from ingestion.utils.logger import logger
from ingestion.utils.storage import DataStorage

# Source paths
csv_path = \
    "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Telco-Customer-Churn.csv"
api_url = "https://my.api.mockaroo.com/users"
# Output path
raw_path = \
    "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset/Raw Data"

storage_path = \
    "/Users/akash/Projects/BITS/DMML/Customer Churn Prediction/Dataset"
# "../Dataset"

API_HEADERS = {"X-API-Key": "2a258740"}

storage = DataStorage(name="Customer Churn Data", storage_root=storage_path)

def ingest_api(api_url, headers, output_dir ):
    logger.info("Starting ingestion")
    _, file = APIDataIngestion.ingest(api_url, output_dir=output_dir, headers=headers)
    logger.info("ingestion Complete")
    logger.info("Starting Data Segregation")
    storage.store(file)
    logger.info("Data Segregation complete")

def ingest_csv(csv_path, output_dir ):
    logger.info("Starting ingestion")
    _, file = CSVDataIngestion.ingest(csv_path, output_dir)
    logger.info("ingestion Complete")
    logger.info("Starting Data Segregation")
    storage.store(file)
    logger.info("Data Segregation complete")

