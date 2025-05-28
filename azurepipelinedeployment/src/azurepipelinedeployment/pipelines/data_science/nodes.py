"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
from lightgbm import LGBMRegressor

def sarimax(daily_occupancy: pd.DataFrame) -> pd.DataFrame:

    """
    Train a SARIMAX model using the daily occupancy data.
    It also logs the using MLflow so it can be used later for predictions.
    Args:
        daily_occupancy (pd.DataFrame): DataFrame containing daily occupancy data with columns:
            - 'ocupacion': Daily occupancy values.
            - 'dia_festivo': Binary indicator for holidays.
            - 'lag_1': Occupancy of the previous day.
            - 'lag_2': Occupancy of two days before.
            - 'lag_4': Occupancy of four days before.
    Returns:
        None: The function logs the model using MLflow.
    """

    y = daily_occupancy['ocupacion']
    exog_cols = ['dia_festivo', 'lag_1', 'lag_2', 'lag_4']
    X = daily_occupancy[exog_cols]

    order = (0, 0, 2) # p, d, q
    seasonal_order = (1, 0, 1, 7)

    modelo_exog = SARIMAX(
    y,
    exog=X,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
    )

    modelo_exog_fit = modelo_exog.fit(disp=False)
    input_example = X.tail(5)
    signature = infer_signature(input_example, modelo_exog_fit.forecast(steps=5, exog=input_example))

    with mlflow.start_run(run_name="train_evaluate_sarimax_model"):

        mlflow.log_param("order", order)
        mlflow.log_param("seasonal_order", seasonal_order)
        mlflow.log_param("enforce_stationarity", False)
        mlflow.log_param("enforce_invertibility", False)
        mlflow.statsmodels.log_model(modelo_exog_fit, "model",
                        registered_model_name="SARIMAX",
                        signature=signature,
                        input_example=input_example)

def lightgbm_pca(daily_demand: pd.DataFrame, feature_cols: list) -> pd.DataFrame:

    """
    Train a LightGBM model using the daily demand data.
    It also logs the model using MLflow so it can be used later for predictions.
    Args:
        daily_demand (pd.DataFrame): DataFrame containing daily demand data with columns:
        featire_cols (list): List of feature columns to be used for training the model.
    Returns:
        pd.DataFrame: DataFrame containing the mean absolute error per platillo_id.
    """

    X = daily_demand[feature_cols]
    y = daily_demand['cantidad']

    params = {'objective':'tweedie','metric':'rmse','verbosity':-1, "bagging_fraction": 0.8,
                "feature_fraction": 0.8,
                "lambda_l1": 1,
                "lambda_l2": 0,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 31,
                }

    model = LGBMRegressor(**params)
    model.fit(
        X,
        y,
        eval_metric='rmse'
    )

    preds = model.predict(X)
    preds = np.floor(preds)

    input_example_df = X.head(5)
    X['cantidad'] = y.values
    X['y_pred'] = preds

    mae_per_plt = (
    X
    .groupby('platillo_id')
    .apply(lambda g: np.floor(mean_absolute_error(g['cantidad'], g['y_pred'])))
    .to_frame(name="mae_per_plt")
    .reset_index()
    )

    preds_df = pd.DataFrame(preds, columns=["cantidad"])
    signature = infer_signature(input_example_df, preds_df)

    with mlflow.start_run(run_name="train_evaluate_lightgbm_model"):
        mlflow.log_params({f"data.{k}": v for k, v in params.items()})
        mlflow.lightgbm.log_model(
        lgb_model = model,
        artifact_path="model",
        signature=signature,
        input_example = input_example_df,
        registered_model_name = "LightGBM"
            )


    return mae_per_plt


















