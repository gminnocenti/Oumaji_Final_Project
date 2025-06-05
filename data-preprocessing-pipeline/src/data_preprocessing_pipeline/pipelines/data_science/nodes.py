"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor

def train_test_split_occupancy(df_occupancy: pd.DataFrame) -> tuple:

    """
    Splits the occupancy DataFrame into training and testing sets.
    The training set consists of all data except the last 30 days, which are used for testing.

    Args:
        df_occupancy (pd.DataFrame): DataFrame containing occupancy data.
    Returns:
        tuple: Contains training and testing sets for features (X_train, X_test) and target variable (y_train, y_test).
    """

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

    """
    Fits a SARIMAX model to the training data and forecasts the next 30 days.
    It also logs the model and its metrics using MLflow.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.
    Returns:
        pd.DataFrame: DataFrame containing the true and predicted occupancy values for the test set.
    """

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
    input_example = X_test.head(1)

    predicciones_exog = np.ceil(predicciones_exog)
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

    with mlflow.start_run():

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

def train_test_split_demand(df_demand: pd.DataFrame) -> tuple:

    """
    Splits the demand DataFrame into training and testing sets.
    The training set consists of all data except the last 30 days, which are used for testing.

    It also generates PCA embeddings for each 'platillo_id' based on the following statistics:
     - Mean
     - Standard Deviation
     - Variance
     - Coefficient of Variation
     - Spike Count
     - Zero Demand Days Ratio

     These features are calculated only using the training data to avoid data leakage.
     Then PCA is applied to reduce the dimensionality of the features.
     The amount of components is determined by the cumulative variance explained (95%)
     And the resulting components are added to the training and testing sets.

     Args:
        df_demand (pd.DataFrame): DataFrame containing demand data.
    Returns:
        tuple: Contains training and testing sets for features (X_train, X_test) and target variable (y_train, y_test).
    """

    df_demand['dia_festivo'].astype('category')
    df_demand['dia_semana'].astype('category')
    df_demand.drop('monto_total', axis=1, inplace=True)
    horizon = 30
    test_df = df_demand.groupby('platillo_id').tail(horizon)
    train_df = df_demand.drop(test_df.index)

    stats = []
    for pid, g in train_df.groupby('platillo_id'):
        s = g.sort_values('fecha')['cantidad']
        stats.append({
            'platillo_id':    pid,
            'mean_sales':     s.mean(),
            'std_sales':      s.std(),
            'var_sales':      s.var(),
            'cv_sales':       s.std()/s.mean() if s.mean() else 0,
            'spike_count':    (s > s.mean() + 2*s.std()).sum(),
            'zero_days_ratio':(s == 0).mean(),
        })
    stats_df = pd.DataFrame(stats)

    num_cols = ['mean_sales','std_sales','var_sales','cv_sales','spike_count','zero_days_ratio']
    scaler = StandardScaler().fit(stats_df[num_cols].fillna(0))
    X_train_stats = scaler.transform(stats_df[num_cols].fillna(0))

    pca_dummy = PCA().fit(X_train_stats)
    cum_var = np.cumsum(pca_dummy.explained_variance_ratio_)
    k = np.searchsorted(cum_var, 0.95) + 1
    pca = PCA(n_components=k, random_state=42).fit(X_train_stats)
    emb_train = pca.transform(X_train_stats)

    emb_df_train = pd.DataFrame(
        emb_train,
        columns=[f'pca_emb_{i}' for i in range(k)]
    )
    emb_df_train['platillo_id'] = stats_df['platillo_id']

    train_df = train_df.merge(emb_df_train, on='platillo_id')
    test_df = test_df.merge(emb_df_train, on='platillo_id')

    feature_cols = ['platillo_id','lag_1','lag_7','ocupacion','dia_semana','dia_festivo'] + [f'pca_emb_{i}' for i in range(k)]

    X_train_demand = train_df[feature_cols]
    y_train_demand = train_df['cantidad']
    X_test_demand  = test_df[feature_cols]
    y_test_demand  = test_df['cantidad']

    return X_train_demand, X_test_demand, y_train_demand, y_test_demand, horizon

def lightgbm_pca(X_train_demand: pd.DataFrame, y_train_demand: pd.Series, X_test_demand: pd.DataFrame, y_test_demand: pd.Series, occupancy_results: pd.DataFrame, horizon: int, dishes_mapping: pd.DataFrame) -> pd.DataFrame:

    """
    Train a LighGBM Model and test it using the occupancy results from the SARIMAX model.
    It also generates intervals for the predictions using the MAE of the training set of each 'platillo_id'.
    It also logs the model and its metrics using MLflow.
    Args:
        X_train_demand (pd.DataFrame): Training features.
        y_train_demand (pd.Series): Training target variable.
        X_test_demand (pd.DataFrame): Testing features.
        y_test_demand (pd.Series): Testing target variable.
        occupancy_results (pd.DataFrame): Results from the SARIMAX model.
        horizon (int): Number of days to forecast.
        dishes_mapping (pd.DataFrame): Mapping of dish IDs to dish names.
    Returns:
        pd.DataFrame: DataFrame containing the true and predicted demand values for the test set.
    """

    occup_vals = occupancy_results['y_pred'].values
    n_platillos = X_test_demand['platillo_id'].nunique()
    X_test_demand['ocupacion'] = np.tile(occup_vals, n_platillos)

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
        X_train_demand,
        y_train_demand,
        eval_set=[(X_test_demand, y_test_demand)],
        eval_metric='rmse'
    )

    preds = model.predict(X_test_demand)
    preds = np.floor(preds)
    mae  = mean_absolute_error(y_test_demand, preds)
    rmse = (root_mean_squared_error(y_test_demand, preds))
    print(f"MAE en últimos {horizon} días: {mae:.2f}")
    print(f"RMSE en últimos {horizon} días: {rmse:.2f}")

    preds_train = model.predict(X_train_demand)
    dummy_df = X_train_demand.copy()
    dummy_df['y_true'] = y_train_demand.values
    dummy_df['y_pred'] = preds_train

    mae_per_plt = (
    dummy_df
    .groupby('platillo_id')
    .apply(lambda g: np.floor(mean_absolute_error(g['y_true'], g['y_pred'])))
    )

    results_demand = X_test_demand.copy()
    results_demand['cantidad'] = y_test_demand.values
    results_demand['pred'] = preds
    results_demand['mae_platillo'] = results_demand['platillo_id'].map(mae_per_plt)

    results_demand['lower'] = results_demand['pred'] - results_demand['mae_platillo']
    results_demand['upper'] = results_demand['pred'] + results_demand['mae_platillo']


    # Mlflow signature
    preds_df = pd.DataFrame(preds_train, columns=["cantidad"])
    signature = infer_signature(X_train_demand, preds_df)

    results_demand.drop(['lag_1','lag_7','dia_semana','dia_festivo','pca_emb_0','pca_emb_1','pca_emb_2'], axis = 1, inplace = True)

    results_demand = results_demand.merge(dishes_mapping, on='platillo_id', how='left')
    dates = occupancy_results['fecha'].values
    n_platillos = results_demand['platillo_id'].nunique()
    results_demand['fecha'] = np.tile(dates, n_platillos)
    results_demand.drop('platillo_id', axis=1, inplace=True)



    with mlflow.start_run():
        mlflow.log_params({f"data.{k}": v for k, v in params.items()})
        mlflow.log_metrics({
        "MAE": mae,
        "RMSE": rmse
            })
        mlflow.lightgbm.log_model(
        lgb_model = model,
        artifact_path="model",
        signature=signature,
        input_example = X_train_demand,
        registered_model_name = "LightGBM"
            )


    return results_demand


















