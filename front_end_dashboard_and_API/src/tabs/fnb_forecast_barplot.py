from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from tabs.make_predictions import prediction_preprocessing,get_prediction_df
import sys
import os
def fnb_forecast_barplot_logic(df_demand: pd.DataFrame,y_pred_30:pd.DataFrame,df_ids:pd.DataFrame,price_df:pd.DataFrame):
    """
    Lógica para el tab de pronóstico de platillos (gráfico de barras).

    Args:
        food_beverage_forecast_data: Datos pronosticados de F&B.
    """
    st.subheader("🍽️ Pronóstico de consumo de platillos de los siguientes 30 días")
    df_demand = df_demand.merge(df_ids, how='left', on='platillo_id')
        # Seleccionador de platillos
   
    all_fc_dishes = df_demand["platillo_cve"].unique()

    default_dishes = ["BBP002", "BBP008", "BBP010", "BBP014"]
    sel_fc_dishes = st.multiselect(
        "Selecciona platillos de tu interés",
        options=all_fc_dishes,
        default=[dish for dish in default_dishes if dish in all_fc_dishes],  # ensure safe default
        placeholder="Busca y selecciona platillos…",
        key="fc_dish",
    )
    if sel_fc_dishes:
        if st.button("Predict"):
            cvpe_to_id = dict(zip(df_ids['platillo_cve'], df_ids['platillo_id']))
            sel_fc_ids = [cvpe_to_id[cve] for cve in sel_fc_dishes if cve in cvpe_to_id]
            with st.spinner("Running predictions..."):
                # Load cached/preprocessed data
                y_pred_30, emb_lookup, future_dates, buffers, mx_holidays = prediction_preprocessing()
                # Generate predictions for the selected IDs
                df_results = get_prediction_df(
                    y_pred_30=y_pred_30,
                    emb_lookup=emb_lookup,
                    future_dates=future_dates,
                    mx_holidays=mx_holidays,
                    platillos=sel_fc_ids,
                    buffers=buffers
                )

            # Display results
            st.subheader("Prediction Results")
            df_results = df_results.merge(df_ids, how='left', on='platillo_id')
            df_results=df_results.merge(price_df, how='left', on='platillo_id')
            df_results['monto_total']=df_results["cantidad_pred"]*df_results["precio_unitario"]
            df_results.drop(columns=["precio_unitario","platillo_id"], inplace=True, errors='ignore')
            df_results["fecha"]= pd.to_datetime(df_results["fecha"])


            # Filtros y controles
            top_row = st.columns((3, 1), gap="medium")
            fc_min, fc_max = (
                df_results["fecha"].min(),
                df_results["fecha"].max(),
            )

            top_row = st.columns((3, 1), gap="medium")
            with top_row[0]:
                fc_range = st.slider(
                    "Forecast window",
                    min_value=fc_min.date(),
                    max_value=fc_max.date(),
                    value=(fc_min.date(), fc_max.date()),
                    step=timedelta(days=1),
                    format="YYYY-MM-DD",
                    key="fc_date",
                )
            with top_row[1]:
                fc_metric = st.radio(
                    "Metric",
                    options=("Unidades", "Ganancias"),
                    horizontal=True,
                    key="fc_metric",
                )



            # Filtrado de datos segun rango de fechas y platillos seleccionados
            start_fc, end_fc = map(pd.Timestamp, fc_range)
            fc_filt = df_results[
                (df_results["fecha"].between(start_fc, end_fc))
                & (df_results["platillo_cve"].isin(sel_fc_dishes))
            ]

            # Métricas atractivas para TCA
            m = st.columns(2)
            m[0].metric(
                "Unidades pronosticadas a vender", f"{int(fc_filt['cantidad_pred'].sum()):,}"
            )
            m[1].metric("Ganancias pronosticadas", f"${fc_filt['monto_total'].sum():,.0f} MXN")

            # Agregaciones por platillo
            if fc_metric == "Unidades":
                fc_agg = (
                    fc_filt.groupby("platillo_cve")["cantidad_pred"]
                    .sum()
                    .sort_values(ascending=False)  # ← fixed
                )
                y_title = "Unidades"
            else:
                fc_agg = (
                    fc_filt.groupby("platillo_cve")["monto_total"]
                    .sum()
                    .sort_values(ascending=False)  # ← fixed
                )
                y_title = "Ganancias (MXN)"
            # Asegurarse de que fc_agg sea un DataFram
                            # Top N platillos pronosticados
            if len(fc_agg) < 5:
                raw_sizes = [len(fc_agg)]
            elif len(fc_agg) > 5:
                raw_sizes = [5,len(fc_agg)]
            elif len(fc_agg) > 10:
                raw_sizes = [5,10,len(fc_agg)]
            sizes = sorted(set(raw_sizes))
            labels = [("Todos" if n == len(fc_agg) else str(n)) for n in sizes]
            label_to_val = dict(zip(labels, sizes))
            chosen_label = st.selectbox(
                "Mostrar top …",
                labels,
                index=labels.index("All") if "All" in labels else len(labels) - 1,
                key="fc_top_n",
            )
            top_n = label_to_val[chosen_label]
            fc_slice = fc_agg.head(top_n)
            # --- layout (chart + dataframe) -------------------------------
            chart_col, table_col = st.columns((3, 2), gap="large")

            # horizontal bar
            with chart_col:
                fig_fc = go.Figure(
                    go.Bar(
                        x=fc_slice.values,
                        y=fc_slice.index.astype(str),
                        orientation="h",
                        text=[
                            f"${v:,.0f}" if fc_metric == "Ganancias" else f"{int(v):,d}"
                            for v in fc_slice.values
                        ],
                        textposition="auto",
                    )
                )
                fig_fc.update_yaxes(autorange="reversed")
                fig_fc.update_layout(
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=10, r=0, t=25, b=10),
                    xaxis_title=y_title,
                    yaxis_title="ID de platillo",
                    title=f"Top {top_n if top_n!=len(fc_agg) else 'todos'} platillos pronosticados por: {fc_metric}",
                )
                st.plotly_chart(fig_fc, use_container_width=True)

            # Dataframe con predicciones de ventas
            # TODO: posiblemente agregar un botón para agrupar los platillos
            # por 'fc_metric', para que sepan de los pronósicos de ventas y ganancias.
            with table_col:
                disp_cols = [
                    "fecha",
                    "platillo_cve",
                    "cantidad_pred" if fc_metric == "Unidades" else "profit",
                ]
                df_disp = fc_filt[disp_cols].sort_values(["fecha", "platillo_cve"])
                st.dataframe(
                    df_disp,
                    hide_index=True,
                    height=450,
                    column_config={
                        "cantidad_pred": st.column_config.NumberColumn(format="%d"),
                        "monto_total": st.column_config.NumberColumn(format="$%.0f"),
                    },
                )
