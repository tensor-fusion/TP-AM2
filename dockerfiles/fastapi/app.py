import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal, Optional
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.
    """
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        with open("/app/files/model.pkl", "rb") as file_ml:
            model_ml = pickle.load(file_ml)
        version_model_ml = 0

    try:
        s3 = boto3.client("s3")

        s3.head_object(Bucket="data", Key="data_info/data.json")
        result_s3 = s3.get_object(Bucket="data", Key="data_info/data.json")
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(
            data_dictionary["standard_scaler_mean"]
        )
        data_dictionary["standard_scaler_std"] = np.array(
            data_dictionary["standard_scaler_std"]
        )
    except:
        with open("/app/files/data.json", "r") as file_s3:
            data_dictionary = json.load(file_s3)

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.
    """
    global model
    global data_dict
    global version_model

    try:
        model_name = "bigmart_sales_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri("http://mlflow:5000")
        client = mlflow.MlflowClient()

        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        if new_version_model != version_model:
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        pass


class ModelInput(BaseModel):
    Item_Weight: Optional[float] = Field(None, description="Weight of the item")
    Item_Visibility: float = Field(description="Visibility of the item in the store")
    Item_MRP: float = Field(description="Maximum Retail Price of the item")
    Outlet_Establishment_Year: int = Field(
        description="Year the outlet was established"
    )
    Outlet_Size: Optional[int] = Field(None, description="Size of the outlet")
    Outlet_Location_Type: int = Field(description="Location type of the outlet")
    Outlet_Type: Literal["Supermarket Type1", "Supermarket Type2"] = Field(
        description="Type of the outlet"
    )

    class Config:
        schema_extra = {
            "example": {
                "Item_Weight": 9.3,
                "Item_Visibility": 0.016047,
                "Item_MRP": 249.8092,
                "Outlet_Establishment_Year": 1999,
                "Outlet_Size": 1,
                "Outlet_Location_Type": 0,
                "Outlet_Type": "Supermarket Type2",
            }
        }


class ModelOutput(BaseModel):
    prediction: float = Field(description="Predicted sales")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 961.4152,
            }
        }


model, version_model, data_dict = load_model("bigmart_sales_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the BigMart Sales Predictor API.
    """
    return JSONResponse(
        content=jsonable_encoder(
            {"message": "Welcome to the BigMart Sales Predictor API"}
        )
    )


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[ModelInput, Body(embed=True)],
    background_tasks: BackgroundTasks,
):
    """
    Endpoint for predicting sales.
    """

    features_df = pd.DataFrame([features.dict()])
    processed_df = pd.DataFrame()

    numerical_features = ["Item_Weight", "Item_Visibility"]
    for feature in numerical_features:
        if feature in features_df.columns:
            processed_df[feature] = (
                features_df[feature]
                - data_dict["standard_scaler_mean"][numerical_features.index(feature)]
            ) / data_dict["standard_scaler_std"][numerical_features.index(feature)]
        else:
            processed_df[feature] = 0

    if "Item_MRP" in features_df.columns:
        processed_df["Item_MRP"] = pd.cut(
            features_df["Item_MRP"], bins=4, labels=[1, 2, 3, 4]
        ).astype(int)
    else:
        processed_df["Item_MRP"] = 0

    if "Outlet_Establishment_Year" in features_df.columns:
        processed_df["Outlet_Establishment_Year"] = features_df[
            "Outlet_Establishment_Year"
        ]
    else:
        processed_df["Outlet_Establishment_Year"] = 0

    if "Outlet_Size" in features_df.columns:
        processed_df["Outlet_Size"] = features_df["Outlet_Size"]
    else:
        processed_df["Outlet_Size"] = 0

    if "Outlet_Location_Type" in features_df.columns:
        processed_df["Outlet_Location_Type"] = features_df["Outlet_Location_Type"]
    else:
        processed_df["Outlet_Location_Type"] = 0

    if "Outlet_Type" in features_df.columns:
        outlet_type_encoded = pd.get_dummies(
            features_df["Outlet_Type"], prefix="Outlet_Type"
        )
        processed_df = pd.concat([processed_df, outlet_type_encoded], axis=1)
    else:
        processed_df["Outlet_Type_Supermarket_Type1"] = 0
        processed_df["Outlet_Type_Supermarket_Type2"] = 0

    expected_columns = data_dict["columns"]
    processed_df = processed_df.reindex(columns=expected_columns, fill_value=0)

    prediction = model.predict(processed_df)

    background_tasks.add_task(check_model)
    return ModelOutput(prediction=float(prediction[0]))
