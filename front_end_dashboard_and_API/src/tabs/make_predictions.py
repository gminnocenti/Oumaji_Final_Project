import pandas as pd
import numpy as np
import streamlit as st
from collections import deque
import holidays
from pandas.tseries.offsets import DateOffset
import requests
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
import os
import sys
import os
from load_data import load_demand,load_occupancy

vault_url = os.environ["KEYVAULT_URI"]
credential = DefaultAzureCredential()
secret_client     = SecretClient(vault_url=vault_url, credential=credential)




@st.cache_data(show_spinner=False)
def prediction_preprocessing() :
    """
    Prepare input data structures for prediction:
      - Download demand and occupancy prediction CSVs.
      - Build embedding lookup for platillo_id.
      - Generate future date range and initialize buffers for lag features.
      - Load Mexican holidays for the future period.
    Returns:
        y_pred_30 (pd.DataFrame): Occupancy predictions indexed by date.
        emb_lookup (pd.DataFrame): PCA embeddings indexed by platillo_id.
        future_dates (pd.DatetimeIndex): Next 30 days for which to predict.
        buffers (dict): Queues of last 7 demand values per platillo_id.
        mx_holidays (holidays.Mexico): Holiday calendar for date checks.
    """
    # Download the main demand dataset
    df_demand = load_demand()
    # Download the precomputed occupancy predictions
    y_pred_30 = load_occupancy()
    # Parse dates and set the index for quick lookup
    #y_pred_30['fecha'] = pd.to_datetime(y_pred_30['fecha'])
    #y_pred_30.set_index('fecha', inplace=True)
    
    # Extract unique PCA embeddings per platillo_id
    emb_df = df_demand[['platillo_id','pca_emb_0','pca_emb_1','pca_emb_2']]
    emb_unique = emb_df.drop_duplicates(subset='platillo_id').reset_index(drop=True)
    emb_unique.index = emb_unique.platillo_id
    emb_lookup = emb_unique.drop(columns='platillo_id')

    # Determine the forecast horizon and last available date
    h = 30
    last_date = pd.to_datetime(df_demand["fecha"].iloc[-1])

    # Create date range for the next h days
    future_dates = pd.date_range(
        start=last_date + DateOffset(days=1),
        periods=h,
        freq='D'
    )

    # Initialize a rolling buffer (deque) of size 7 for each platillo_id
    buffers = {}
    for pid, group in df_demand.sort_values("fecha").groupby("platillo_id"):
        last7 = group["cantidad"].tail(7).tolist()
        buffers[pid] = deque(last7, maxlen=7)

    # Load the Mexican holiday calendar for the relevant years
    mx_holidays = holidays.Mexico(
        years=range(future_dates[0].year, future_dates[-1].year + 1)
    )
    return y_pred_30, emb_lookup, future_dates, buffers, mx_holidays

@st.cache_resource
def get_prediction_client():
    # grab endpoint/url/api key only once per app session
    endpoint = secret_client.get_secret("sarimax-endpoint-url").value
    api_key  = secret_client.get_secret("sarimax-api-key").value

    # use a single Session for connection‐pooling
    sess = requests.Session()
    sess.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    })
    return endpoint, sess
def model_make_prediction(df_input: pd.DataFrame) -> dict:
    endpoint, sess = get_prediction_client()
    payload = { "input_data": { "columns": df_input.columns.tolist(),
                                "data":    df_input.values.tolist() } }
    resp = sess.post(endpoint, json=payload)
    resp.raise_for_status()
    return resp.json()


def get_prediction_df(
    y_pred_30: pd.DataFrame,
    emb_lookup: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    mx_holidays,
    platillos: list,
    buffers: dict
) -> pd.DataFrame:
    """
    Iterate over each future date and platillo to generate demand predictions.

    Args:
        y_pred_30 (pd.DataFrame): Occupancy data for feature input.
        emb_lookup (pd.DataFrame): Embeddings keyed by platillo_id.
        future_dates (pd.DatetimeIndex): Dates to forecast.
        mx_holidays (holidays.Mexico): Holiday lookup.
        platillos (list): List of platillo_id values to predict for.
        buffers (dict): Rolling demand buffers per platillo_id.
    Returns:
        pd.DataFrame: Combined DataFrame of date, platillo_id, and predicted demand.
    """
    # Create a new DataFrame for the predictions
    pred_rows = []

    #platillos=df_demand["platillo_id"].unique()
    for fecha in future_dates:                          
        festivo     = int(fecha.normalize() in mx_holidays)
        dia_semana  = fecha.weekday()
        ocupacion_d = y_pred_30.loc[fecha,"ocupacion_pred"]              

        for pid in platillos:                           
            dq = buffers[pid]
            lag_1 = dq[-1]
            lag_7 = dq[0]

            row_dict = {
                "platillo_id": pid,
                "lag_1":       lag_1,
                "lag_7":       lag_7,
                "ocupacion":   ocupacion_d,
                "dia_semana":  dia_semana,
                "dia_festivo": festivo,
                **emb_lookup.loc[pid].to_dict(),
            }
            df_input = pd.DataFrame(row_dict, index=[0])

            #print(df_input)
            # 2) call your endpoint
            result_json = model_make_prediction(df_input)

            # 3) extract the numeric prediction
            #    adapt the key below to whatever your endpoint returns;
            #    common patterns are "result" or "predictions"
            if isinstance(result_json, list):
                # e.g. [27.2401…]
                y_hat = float(result_json[0])
            elif isinstance(result_json, dict) and "result" in result_json:
                y_hat = float(result_json["result"][0][0])
            elif isinstance(result_json, dict) and "predictions" in result_json:
                y_hat = float(result_json["predictions"][0][0])
            else:
                raise KeyError(f"Couldn't find prediction in {result_json}")
            pred_rows.append({"fecha": fecha, "platillo_id": pid, "cantidad_pred": np.floor(y_hat)})
            dq.append(y_hat)

    df_pred_demand = (
        pd.DataFrame(pred_rows)
        .sort_values(["platillo_id", "fecha"]) 
        .reset_index(drop=True)
    )
    
    return df_pred_demand

