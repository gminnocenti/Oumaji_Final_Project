"""Módulo para cargar y preparar datos de CSVs para Streamlit.
Se utiliza cache para que no se tenga que cargar cada vez que se actualiza la app."""

import streamlit as st
import pandas as pd
from pathlib import Path


import io
import os
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import numpy as np
from azure.storage.blob import BlobServiceClient

import streamlit as st
from dotenv import load_dotenv
load_dotenv()
vault_url = os.environ["KEYVAULT_URI"]
credential = DefaultAzureCredential()
secret_client     = SecretClient(vault_url=vault_url, credential=credential)
# Authenticate using DefaultAzureCredential, which supports managed identity or local development



################################################ Functions to connect to Azure Blob Storage and download datasets
def get_storage_connection_string() -> str:
    """
    Fetch storage account name and key from Key Vault and build the connection string.
    """
    # Retrieve the storage account name and key from Key Vault secrets
    account_name = secret_client.get_secret("storage-account-name").value
    account_key = secret_client.get_secret("storage-account-key").value

    # Format the Azure Blob Storage connection string
    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )
    return conn_str

@st.cache_resource(show_spinner=False)
def get_blob_service() -> BlobServiceClient:
    conn_str = get_storage_connection_string()
    return BlobServiceClient.from_connection_string(conn_str)

def download_blob_storage_df(
    blob_storage_name: str,
    dataset_directory: str,
) -> pd.DataFrame:
    """
    Download a blob from Azure Blob Storage and load it into a DataFrame.

    Args:
        blob_storage_name (str): Name of the blob container.
        dataset_directory (str): Path to the blob within the container.

    Returns:
        pd.DataFrame: DataFrame with the contents of the CSV blob.
    """
    # Build the connection string and instantiate the BlobServiceClient
    blob_service = get_blob_service()
    
    # Get a client for the target container and blob
    container_client = blob_service.get_container_client(blob_storage_name)
    blob_client = container_client.get_blob_client(dataset_directory)
    
    # Download the blob content into memory
    stream = blob_client.download_blob().readall()
    
    # Read the CSV bytes into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(stream))
    return df

@st.cache_data(show_spinner=False)
def load_pricedf() -> pd.DataFrame:
    return download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="08_reporting/price_df.csv"
    )
@st.cache_data(show_spinner=False)
def load_dishes_mapping() -> pd.DataFrame:
    return download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="03_primary/dishes_mapping.csv"
    )
@st.cache_data(show_spinner=False)
def load_demand() -> pd.DataFrame:
    return download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="03_primary/daily_demand.csv"
    )
@st.cache_data(show_spinner=False)
def load_occupancy() -> pd.DataFrame:
    df = download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="08_reporting/occupancy_predictions.csv"
    )
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.set_index('fecha')

@st.cache_data(show_spinner=False)
def load_occupancy_forecast() -> pd.DataFrame:
    df = download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="08_reporting/occupancy_predictions.csv"
    )
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

@st.cache_data(show_spinner=False)
def load_occupancy_csv() -> pd.DataFrame:
    """Lee y cachea el CSV especificado."""
    df = download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="03_primary/daily_occupancy.csv"
    )
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


@st.cache_data(show_spinner=False)
def load_food_beverage_csv(
    csv_path: str | Path, parse_dates: list[str]
) -> pd.DataFrame:
    """Lee y cachea el CSV de F&B especificado."""
    return pd.read_csv(csv_path, parse_dates=parse_dates)
@st.cache_data(show_spinner=False)
def load_demand_pred() -> pd.DataFrame:
    """Lee y cachea el CSV especificado."""
    df = download_blob_storage_df(
        blob_storage_name="tca-blob-storage",
        dataset_directory="08_reporting/demand_predictions.csv"
    )
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df
