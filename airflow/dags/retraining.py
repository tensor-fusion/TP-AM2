import datetime
from airflow.decorators import dag, task

markdown_text = """
### Re-Train the Model for BigMart Sales Data

This DAG re-trains the model based on new data, tests the previous model, and puts in production the new one 
if it performs better than the old one. It uses the RMSE to evaluate the model with the test data.
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
    dag_id="retrain_the_model_bigmart",
    description="Re-train the model based on new data, tests the previous model, and put in production the new one if "
    "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Re-Train", "BigMart"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=[
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "awswrangler==3.6.0",
            "boto3",
        ],
        system_site_packages=True,
    )
    def train_the_challenger_model():
        import datetime
        import json
        import boto3
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error
        from mlflow.models import infer_signature
        from sklearn.preprocessing import StandardScaler

        mlflow.set_tracking_uri("http://mlflow:5000")

        def preprocess_data(X):
            print("Input columns:", X.columns.tolist())
            print("\nItem_MRP statistics:")
            print(X["Item_MRP"].describe())
            print("\nItem_MRP unique values:", X["Item_MRP"].nunique())
            print("\nItem_MRP value counts:\n", X["Item_MRP"].value_counts().head())

            X["Item_MRP"] = pd.cut(X["Item_MRP"], bins=4, labels=[1, 2, 3, 4])

            columns_to_drop = ["Item_Type", "Item_Fat_Content"]
            existing_columns = [col for col in columns_to_drop if col in X.columns]
            if existing_columns:
                X = X.drop(columns=existing_columns)

            if "Outlet_Size" in X.columns:
                X["Outlet_Size"] = X["Outlet_Size"].replace(
                    {"High": 2, "Medium": 1, "Small": 0}
                )
            if "Outlet_Location_Type" in X.columns:
                X["Outlet_Location_Type"] = X["Outlet_Location_Type"].replace(
                    {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}
                )

            if "Outlet_Type" in X.columns:
                X = pd.get_dummies(X, columns=["Outlet_Type"], drop_first=True)

            scaler = StandardScaler()
            numerical_features = [
                col for col in ["Item_Weight", "Item_Visibility"] if col in X.columns
            ]
            if numerical_features:
                X[numerical_features] = scaler.fit_transform(X[numerical_features])

            print("\nOutput columns:", X.columns.tolist())
            return X

        def load_the_champion_model():
            model_name = "bigmart_sales_model_prod"
            alias = "champion"
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)
            champion_version = mlflow.sklearn.load_model(model_data.source)
            return champion_version

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/bigmart_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/bigmart_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/bigmart_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/bigmart_y_test.csv")
            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            experiment = mlflow.set_experiment("BigMart Sales")
            mlflow.start_run(
                run_name="Challenger_run_"
                + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "BigMart"},
                log_system_metrics=True,
            )
            params = model.get_params()
            params["model"] = type(model).__name__
            mlflow.log_params(params)
            artifact_path = "model"
            signature = infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format="cloudpickle",
                registered_model_name="bigmart_sales_model_dev",
                metadata={"model_data_version": 1},
            )
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, rmse, model_uri):
            client = mlflow.MlflowClient()
            name = "bigmart_sales_model_prod"
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["rmse"] = rmse
            result = client.create_model_version(
                name=name, source=model_uri, run_id=model_uri.split("/")[-3], tags=tags
            )
            client.set_registered_model_alias(name, "challenger", result.version)

        champion_model = load_the_champion_model()
        challenger_model = clone(champion_model)
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        X_train_processed = preprocess_data(X_train)
        X_test_processed = preprocess_data(X_test)

        challenger_model.fit(X_train_processed, y_train.to_numpy().ravel())
        y_pred = challenger_model.predict(X_test_processed)
        rmse = mean_squared_error(y_test.to_numpy().ravel(), y_pred, squared=False)

        artifact_uri = mlflow_track_experiment(challenger_model, X_train_processed)
        register_challenger(challenger_model, rmse, artifact_uri)

    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=[
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "awswrangler==3.6.0",
            "boto3",
        ],
        system_site_packages=True,
    )
    def evaluate_champion_challenge():
        import json
        import boto3
        import mlflow
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import StandardScaler

        mlflow.set_tracking_uri("http://mlflow:5000")

        def preprocess_data(X):
            print("Input columns:", X.columns.tolist())
            print("\nItem_MRP statistics:")
            print(X["Item_MRP"].describe())
            print("\nItem_MRP unique values:", X["Item_MRP"].nunique())
            print("\nItem_MRP value counts:\n", X["Item_MRP"].value_counts().head())

            X["Item_MRP"] = pd.cut(X["Item_MRP"], bins=4, labels=[1, 2, 3, 4])

            columns_to_drop = ["Item_Type", "Item_Fat_Content"]
            existing_columns = [col for col in columns_to_drop if col in X.columns]
            if existing_columns:
                X = X.drop(columns=existing_columns)

            if "Outlet_Size" in X.columns:
                X["Outlet_Size"] = X["Outlet_Size"].replace(
                    {"High": 2, "Medium": 1, "Small": 0}
                )
            if "Outlet_Location_Type" in X.columns:
                X["Outlet_Location_Type"] = X["Outlet_Location_Type"].replace(
                    {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}
                )

            if "Outlet_Type" in X.columns:
                X = pd.get_dummies(X, columns=["Outlet_Type"], drop_first=True)

            scaler = StandardScaler()
            numerical_features = [
                col for col in ["Item_Weight", "Item_Visibility"] if col in X.columns
            ]
            if numerical_features:
                X[numerical_features] = scaler.fit_transform(X[numerical_features])

            print("\nOutput columns:", X.columns.tolist())
            return X

        def load_the_model(alias):
            model_name = "bigmart_sales_model_prod"
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)
            model = mlflow.sklearn.load_model(model_data.source)
            return model

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/bigmart_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/bigmart_y_test.csv")
            return X_test, y_test

        def promote_challenger(name):
            client = mlflow.MlflowClient()
            client.delete_registered_model_alias(name, "champion")
            challenger_version = client.get_model_version_by_alias(name, "challenger")
            client.delete_registered_model_alias(name, "challenger")
            client.set_registered_model_alias(
                name, "champion", challenger_version.version
            )

        def demote_challenger(name):
            client = mlflow.MlflowClient()
            client.delete_registered_model_alias(name, "challenger")

        champion_model = load_the_model("champion")
        challenger_model = load_the_model("challenger")
        X_test, y_test = load_the_test_data()

        X_test_processed = preprocess_data(X_test)

        y_pred_champion = champion_model.predict(X_test_processed)
        rmse_champion = mean_squared_error(
            y_test.to_numpy().ravel(), y_pred_champion, squared=False
        )

        y_pred_challenger = challenger_model.predict(X_test_processed)
        rmse_challenger = mean_squared_error(
            y_test.to_numpy().ravel(), y_pred_challenger, squared=False
        )

        experiment = mlflow.set_experiment("BigMart Sales")
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_rmse_challenger", rmse_challenger)
            mlflow.log_metric("test_rmse_champion", rmse_champion)
            if rmse_challenger < rmse_champion:
                mlflow.log_param("Winner", "Challenger")
            else:
                mlflow.log_param("Winner", "Champion")

        name = "bigmart_sales_model_prod"
        if rmse_challenger < rmse_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
