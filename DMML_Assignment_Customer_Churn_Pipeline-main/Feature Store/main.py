import datetime
from os import times

import pandas as pd
from pyarrow import timestamp

from utils.retriver import  CustomerChurnFeatures


source_path = "../Dataset/Processed Data/processed_data.csv"


# if __name__ == "__main__":
#     customer_ids = pd.read_csv(source_path)["customerID"].to_list()
#
#     customer_churn_features = CustomerChurnFeatures("./")
#     df = customer_churn_features.load(source_path)
#
#     entity_df = pd.DataFrame({
#         "customerID": customer_ids[:100],
#         "event_timestamp": [datetime.datetime.today() for i in customer_ids[:100]]
#     })
#     vector = customer_churn_features.retrieve(entity_df)
#     print(vector)
