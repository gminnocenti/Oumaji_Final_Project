"""Main logic for streamlit dashboard application."""

from pathlib import Path
import streamlit as st
from PIL import Image
from load_data import load_occupancy_csv,load_occupancy,load_occupancy_forecast,load_demand,load_pricedf,load_dishes_mapping
# Tabs
from tabs.occupancy_tab import occupancy_tab_logic
from tabs.fnb_bi import food_beverage_bi_logic
from tabs.fnb_forecast import fnb_forecast
repo_path = Path(__file__).resolve().parent.parent
web_icon = Image.open("page_icon.webp")
st.set_page_config(page_title="Dashboard TCA", page_icon = web_icon, layout="wide")
st.markdown(
    """
<style>
/* tighten the padding around metrics (flash-cards) */
[data-testid="metric-container"] {
    padding: 10px 15px;
}
</style>
""",
    unsafe_allow_html=True,
)

# === CARGA DE DATOS ===
try:
    # Ocuppancy
    hist = load_occupancy_csv()
    fcst = load_occupancy_forecast()
    # Food & Beverage
    fnb_hist = load_demand()
    df_ids=load_dishes_mapping()
    df_demand = load_demand()
    y_pred_30 = load_occupancy()
    price_df=load_pricedf()

except FileNotFoundError:
    st.error(
        "No se encontraron los archivos."
    )
    st.stop()


# === STREAMLIT UI ===
st.title("OUMAJI MVP Dashboard")

occupancy_tab, fnb_tab_bi, fnb_tab_forecast = st.tabs(
    [
        "📈  Ocupación en el tiempo + predicciones",
        "📊  BI de B&A",
        "🍔  Predicciones de B&A",
        
    ]
)

# === OCCUPANCY ===™
with occupancy_tab:
    occupancy_tab_logic(historic_data=hist, forecast_data=fcst)

# === FOOD & BEVERAGE BI ===
with fnb_tab_bi:
    food_beverage_bi_logic(food_beverage_historic_data=fnb_hist,df_ids=df_ids)

# === FOOD & BEVERAGE FORECAST ===
with fnb_tab_forecast:
    fnb_forecast(df_demand=df_demand,y_pred_30=y_pred_30,df_ids=df_ids,price_df=price_df)
