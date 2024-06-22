import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process for BigMart Sales Data

This DAG extracts information from the provided CSV file for the BigMart Sales dataset. 
It preprocesses the data by encoding categorical variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 80/20.
"""

default_args = {
    "owner": "Milton",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=15),
}


@dag(
    dag_id="process_etl_bigmart_data",
    description="ETL process for BigMart sales data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "BigMart Sales"],
    default_args=default_args,
    catchup=False,
)
def process_etl_bigmart_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def get_data():
        """
        Load the raw data from the local directory
        """
        import awswrangler as wr
        import pandas as pd
        import numpy as np

        data_path_train = "/opt/airflow/dags/data/Train.csv"
        data_path_test = "/opt/airflow/dags/data/Test.csv"

        train_df = pd.read_csv(data_path_train)
        test_df = pd.read_csv(data_path_test)

        dataframe = pd.concat([train_df, test_df], ignore_index=True)

        combined_data_path = "s3://data/raw/bigmart_combined.csv"
        wr.s3.to_csv(df=dataframe, path=combined_data_path, index=False)

    @task.virtualenv(
        task_id="preprocess_data",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2"],
        system_site_packages=True,
    )
    def preprocess_data():
        """
        Preprocess the data according to the new approach.
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        data_original_path = "s3://data/raw/bigmart_combined.csv"
        data_end_path = "s3://data/raw/bigmart_preprocessed.csv"
        dataset = wr.s3.read_csv(data_original_path)

        dataset.drop_duplicates(inplace=True, ignore_index=True)
        dataset.dropna(inplace=True, ignore_index=True)

        numerical_features = ["Item_Weight", "Item_Visibility", "Item_MRP"]
        categorical_features = [
            "Item_Fat_Content",
            "Item_Type",
            "Outlet_Size",
            "Outlet_Location_Type",
            "Outlet_Type",
        ]

        imputer = SimpleImputer(strategy="mean")
        dataset[numerical_features] = imputer.fit_transform(dataset[numerical_features])

        for feature in categorical_features:
            dataset[feature].fillna("Unknown", inplace=True)

        # Bin 'Item_MRP' en cuartiles
        dataset["Item_MRP"] = pd.qcut(dataset["Item_MRP"], 4, labels=[1, 2, 3, 4])

        # Eliminamos variables que no aportan considerablemente a la predicción
        dataset = dataset.drop(columns=["Item_Type", "Item_Fat_Content"])
        dataset = dataset.drop(columns=["Item_Identifier", "Outlet_Identifier"])

        dataset["Outlet_Size"] = dataset["Outlet_Size"].replace(
            {"High": 2, "Medium": 1, "Small": 0, "Unknown": -1}
        )
        dataset["Outlet_Location_Type"] = dataset["Outlet_Location_Type"].replace(
            {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0, "Unknown": -1}
        )

        dataset = pd.get_dummies(dataset, columns=["Outlet_Type"])

        scaler = StandardScaler()
        numerical_features = ["Item_Weight", "Item_Visibility"]
        dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

        wr.s3.to_csv(df=dataset, path=data_end_path, index=False)

        client = boto3.client("s3")

        data_dict = {}
        try:
            client.head_object(Bucket="data", Key="data_info/data.json")
            result = client.get_object(Bucket="data", Key="data_info/data.json")
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise e

        target_col = "Item_Outlet_Sales"
        dataset_log = dataset.drop(columns=[target_col])

        data_dict["columns"] = dataset_log.columns.to_list()
        data_dict["target_col"] = target_col
        data_dict["columns_dtypes"] = {
            k: str(v) for k, v in dataset_log.dtypes.to_dict().items()
        }
        data_dict["standard_scaler_mean"] = scaler.mean_.tolist()
        data_dict["standard_scaler_std"] = scaler.scale_.tolist()

        data_dict["date"] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(Bucket="data", Key="data_info/data.json", Body=data_string)

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("BigMart Sales")

        mlflow.start_run(
            run_name="ETL_run_"
            + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "BigMart sales"},
            log_system_metrics=True,
        )

        mlflow_dataset = mlflow.data.from_pandas(
            dataset,
            source="s3://data/raw/Train.csv",
            targets=target_col,
            name="bigmart_data_preprocessed",
        )
        mlflow.log_input(mlflow_dataset, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2"],
        system_site_packages=True,
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_preprocessed_path = "s3://data/raw/bigmart_preprocessed.csv"
        dataset = wr.s3.read_csv(data_preprocessed_path)

        test_size = Variable.get("test_size_bigmart")
        target_col = Variable.get("target_col_bigmart")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        save_to_csv(X_train, "s3://data/final/train/bigmart_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/bigmart_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/bigmart_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/bigmart_y_test.csv")

    get_data() >> preprocess_data() >> split_dataset()


dag = process_etl_bigmart_data()
