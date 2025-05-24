"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature

def train_test_split_occupancy(df_occupancy: pd.DataFrame) -> tuple:

    df_occupancy = df_occupancy.copy()

    df_occupancy['fecha'] = pd.to_datetime(df_occupancy['fecha'])
    df_occupancy = df_occupancy.sort_values('fecha')
    df_occupancy.set_index('fecha', inplace=True)

    # Variable objetivo
    y = df_occupancy['ocupacion']

    # Variables exógenas
    # X = df.drop(columns=['ocupacion'])
    exog_cols = ['dia_festivo', 'lag_1', 'lag_2', 'lag_4']
    X = df_occupancy[exog_cols]

    horizonte = 30
    y_train = y.iloc[:-horizonte]
    y_test = y.iloc[-horizonte:]
    X_train = X.iloc[:-horizonte]
    X_test = X.iloc[-horizonte:]

    return X_train, X_test, y_train, y_test

def sarimax(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:


    order = (0, 0, 2) # p, d, q
    seasonal_order = (1, 0, 1, 7)

    modelo_exog = SARIMAX(
    y_train,
    exog=X_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
    )

    modelo_exog_fit = modelo_exog.fit(disp=False)

    predicciones_exog = modelo_exog_fit.forecast(steps=30, exog=X_test)
    predicciones_exog.index = y_test.index  # Para que los índices coincidan
    signature = infer_signature(X_train, modelo_exog_fit.forecast(steps=30, exog=X_test))
    input_example = X_test.head(3)

    mae = mean_absolute_error(y_test, predicciones_exog)
    mpe = ((y_test - predicciones_exog) / y_test).abs().mean() * 100
    rmse = root_mean_squared_error(y_test, predicciones_exog)

    print(f"MAE: {mae:.2f}")
    print(f"MPE: {mpe:.2f}%")
    print(f"RMSE: {rmse:.2f}")

    results_occupancy = (
        pd.DataFrame({"y_true": y_test, "y_pred": predicciones_exog})
        .reset_index()
        .rename(columns={"index": "fecha"})
    )

    with mlflow.start_run(run_name="train_evaluate_sarimax_model"):

        mlflow.log_param("order", order)
        mlflow.log_param("seasonal_order", seasonal_order)
        mlflow.log_param("enforce_stationarity", False)
        mlflow.log_param("enforce_invertibility", False)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MPE", mpe)
        mlflow.log_metric("RMSE", rmse)
        mlflow.statsmodels.log_model(modelo_exog_fit, "model",
                        registered_model_name="SARIMAX",
                        signature=signature,
                        input_example=input_example)

    return results_occupancy










