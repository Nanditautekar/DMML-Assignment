from datetime import timedelta, datetime
import pandas as pd
from feast import Feature, FeatureView, Entity, ValueType, FileSource, FeatureStore


class CustomerChurnFeatures:
    def __init__(self, repo_path:str):

        # Initialize the feature store
        self.store = FeatureStore(repo_path=repo_path)
        # Define the customer entity using customerID as the key
        self.entity = Entity(
            name="customerID",
            value_type=ValueType.STRING,  # assuming customerID is a string, adjust if numeric
            description="Unique identifier for each customer"
        )
        self.customer_features_view = None
        self.customer_data_source = None

    def load(self, path):
        df = pd.DataFrame()
        if ".csv" in path:
            df = pd.read_csv(path)
            path = path.split(".csv")[0]+".parquet"
            df["event_timestamp"] = datetime.today()
            df.to_parquet(path)
        # Define a FileSource for the CSV file
        self.customer_data_source = FileSource(
            path=path,
            event_timestamp_column="tenure",  # Using tenure as a placeholder for event timestamp.
            created_timestamp_column=None  # Adjust if you have a created timestamp column.
        )

        # Define a FeatureView for customer churn features.
        # We are including all columns from the CSV except customerID.
        self.customer_features_view = FeatureView(
            name="customer_churn_features_view",
            entities=[self.entity],
            ttl=timedelta(days=1),
            online=True,
            source=self.customer_data_source,
            description="Feature view for customer churn including demographic and usage features. Version: 1.0"
        )
        self.store.apply(self.customer_features_view)
        return df

    def retrieve(self, entity_df:pd.DataFrame=None):

        # Retrieve features for training or inference
        feature_vector = self.store.get_historical_features(
            entity_df=entity_df,
            features=[
                "customer_churn_features_view:customerID",
                "customer_churn_features_view:gender",
                "customer_churn_features_view:SeniorCitizen",
                "customer_churn_features_view:Partner",
                "customer_churn_features_view:Dependents",
                "customer_churn_features_view:tenure",
                "customer_churn_features_view:PhoneService",
                "customer_churn_features_view:MultipleLines",
                "customer_churn_features_view:InternetService",
                "customer_churn_features_view:OnlineSecurity",
                "customer_churn_features_view:OnlineBackup",
                "customer_churn_features_view:DeviceProtection",
                "customer_churn_features_view:TechSupport",
                "customer_churn_features_view:StreamingTV",
                "customer_churn_features_view:StreamingMovies",
                "customer_churn_features_view:Contract",
                "customer_churn_features_view:PaperlessBilling",
                "customer_churn_features_view:PaymentMethod",
                "customer_churn_features_view:MonthlyCharges",
                "customer_churn_features_view:TotalCharges",
                "customer_churn_features_view:Churn"
            ]
        ).to_df()

        return feature_vector


