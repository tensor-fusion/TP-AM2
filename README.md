# MLOps Project - ML II

# Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Services and Access](#services-and-access)
  - [Apache Airflow](#apache-airflow)
  - [MLflow](#mlflow)
  - [MinIO](#minio)
  - [PostgreSQL](#postgresql)
  - [FastAPI](#fastapi)
- [Dataset](#dataset)
- [Model](#model)
  - [ETL Workflow](#etl-workflow)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Model Serving](#model-serving)
  - [Model Retraining](#model-retraining)
- [Deployment](#deployment)
  - [Docker Compose](#docker-compose)
  - [Running the Project](#running-the-project)
- [Dependencies](#dependencies)
- [License](#license)


## Overview

This is an MLOps project built to simulate a real-world scenario in which we predict product sales using ML.

It uses several services including `Apache Airflow` for orchestration, `MLflow` for experiment tracking and model management, `MinIO` for storage, `PostgreSQL` for database management, and `FastAPI` for serving the model.

## Architecture

![Architecture](/images/architecture.png)

The architecture consists of the following components:

- **Jupyter Notebook**: For experimentation and model training.
- **MLflow**: For tracking experiments and managing models.
- **MinIO**: S3-compatible storage for data and models.
- **PostgreSQL**: Database for Airflow and MLflow.
- **Apache Airflow**: For orchestrating the ETL and model retraining workflows.
- **FastAPI**: For serving the trained model via a REST API.

## Services and Access

### Apache Airflow
- **Web Server**: Accessible at `http://localhost:8080`
- **Scheduler**: Manages the DAG runs.

### MLflow
- **Tracking Server**: Accessible at `http://localhost:5000`
- **Model Registry**: Manages the model versions.

### MinIO
- **Console**: Accessible at `http://localhost:9001`
- **Buckets**: 
  - `s3://data`: For storing the raw and processed data.
  - `s3://mlflow`: For storing MLflow artifacts.

### PostgreSQL
- **Airflow Database**: `postgresql://airflow:airflow@localhost:5432/airflow`
- **MLflow Database**: `postgresql://airflow:airflow@localhost:5432/mlflow_db`

### FastAPI
- **API Endpoint**: Accessible at `http://localhost:8800/predict`

## Dataset

The dataset used for this project is the classic `BigMart` Sales dataset. This dataset is widely used for regression tasks in machine learning and contains information about various products sold across multiple outlets. The goal is to predict the sales of each product in different stores.

### Data Fields

The dataset consists of the following columns:

| Feature Name               | Description |
|----------------------------|-------------|
| `Item_Identifier`          | Unique product ID. |
| `Item_Weight`              | Weight of the product. |
| `Item_Fat_Content`         | Indicates whether the product is low fat or regular. Categories include `Low Fat`, `Regular`, and some mislabeled values that need cleaning. |
| `Item_Visibility`          | The percentage of total display area of all products in a store allocated to the particular product. |
| `Item_Type`                | The category to which the product belongs. For example, `Dairy`, `Soft Drinks`, etc. |
| `Item_MRP`                 | Maximum Retail Price (list price) of the product. |
| `Outlet_Identifier`        | Unique identifier of the store. |
| `Outlet_Establishment_Year`| The year in which the store was established. |
| `Outlet_Size`              | The size of the store in terms of the area covered. Categories include `Small`, `Medium`, and `High`. |
| `Outlet_Location_Type`     | The type of city in which the store is located. Categories include `Tier 1`, `Tier 2`, and `Tier 3`. |
| `Outlet_Type`              | The type of the store. Categories include `Grocery Store`, `Supermarket Type1`, `Supermarket Type2`, and `Supermarket Type3`. |
| `Item_Outlet_Sales`        | Sales of the product in the particular store. This is the **target** variable to be predicted. |

## Model

The model used in this project is a regression model, optimized using `Optuna` for hyperparameter tuning. The primary metric used for evaluation is the **RMSE** (Root Mean Squared Error).

### ETL Workflow

The ETL process extracts data from the dataset CSV file, preprocesses it, and saves it into S3 buckets. The process is managed by an `Airflow` DAG.

```python
# ETL DAG snippet
import datetime
from airflow.decorators import dag, task

@dag(dag_id="process_etl_bigmart_data", ...)
def process_etl_bigmart_data():
    @task.virtualenv(task_id="obtain_original_data", ...)
    def get_data():
        # Load and preprocess data

    @task.virtualenv(task_id="preprocess_data", ...)
    def preprocess_data():
        # Further preprocessing

    @task.virtualenv(task_id="split_dataset", ...)
    def split_dataset():
        # Split the data into training and testing sets

    get_data() >> preprocess_data() >> split_dataset()

dag = process_etl_bigmart_data()
```

![etl](/images/etl.png)

### Hyperparameter Tuning

Hyperparameter tuning is done using `Optuna` within a Jupyter Notebook, with results tracked in `MLflow`.

```python
import optuna
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    # Define the objective function for Optuna

study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

# Log best parameters in MLflow
mlflow.log_params(study.best_params)
mlflow.log_metric("best_rmse", study.best_value)
```

### Model Serving

The trained model is served via `FastAPI`, allowing for real-time predictions.

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {        
      "Item_Weight": 9.3,         
      "Item_Visibility": 0.016047,
      "Item_MRP": 249.8092,             
      "Outlet_Establishment_Year": 1999,
      "Outlet_Size": 1,         
      "Outlet_Location_Type": 0,        
      "Outlet_Type": "Supermarket Type2"
    }
  }'
```

```bash
{
   "prediction":1469.1481616393794
}
```

### Model Retraining

A separate `Airflow` DAG handles model retraining. If the new model performs better than the current one, it is promoted to production.

```python
# Retraining DAG snippet
import datetime
from airflow.decorators import dag, task

@dag(dag_id="retrain_the_model_bigmart", ...)
def processing_dag():
    @task.virtualenv(task_id="train_the_challenger_model", ...)
    def train_the_challenger_model():
        # Train the new model

    @task.virtualenv(task_id="evaluate_champion_challenge", ...)
    def evaluate_champion_challenge():
        # Evaluate and promote/demote models

    train_the_challenger_model() >> evaluate_champion_challenge()

my_dag = processing_dag()
```

![etl](/images/retrain.png)

## Deployment

### Docker Compose

The entire project can be deployed using **Docker Compose**. The `docker-compose.yml` file includes definitions for all services.


```yaml

services:
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mlflow:
    image: mlflow:latest
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    ports:
      - "5000:5000"

  airflow:
    image: apache/airflow:latest
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
    ports:
      - "8080:8080"

  fastapi:
    image: fastapi:latest
    ports:
      - "8800:8800"

volumes:
  postgres_data:
```

### Running the Project

1. Clone the repository.
2. Build and start the services using Docker Compose:

```bash
docker-compose --profile all up
```

3. Access the services via their respective URLs.

## Dependencies

- Python 3.8+
- Apache Airflow
- MLflow
- MinIO
- PostgreSQL
- FastAPI
- Docker

## License

This project is licensed under the Apache License.