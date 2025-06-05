from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from make_predictions import prediction_preprocessing, get_prediction_df


def fnb_forecast(
    df_demand: pd.DataFrame,
    y_pred_30: pd.DataFrame,
    df_ids: pd.DataFrame,
    price_df: pd.DataFrame,
    demand_pred: pd.DataFrame
):
    # Preparamos los datos históricos de demanda
    df_demand["fecha"] = pd.to_datetime(df_demand["fecha"])
    df_demand = (
        df_demand
        .merge(df_ids, on="platillo_id", how="left")
        .merge(price_df, on="platillo_id", how="left")
    )
    df_demand["monto_total"] = df_demand["cantidad"] * df_demand["precio_unitario"]

    # Esto nos servirá para guardar las predicciones en session_state y que no se reinicie cada vez
    # que seleccionemos algo
    if "pred_fc" not in st.session_state:
        st.session_state.pred_fc = None
        st.session_state.pred_ids = []

    st.subheader("🍽️ Pronóstico de consumo de platillos de los siguientes 30 días")
    # Selector de platillos
    all_fc_dishes = df_demand["platillo_cve"].unique()
    default_dishes = ["BBP002", "BBP008", "BBP010", "BBP014"]
    sel_fc_dishes = st.multiselect(
        "Selecciona platillos de tu interés",
        options=all_fc_dishes,
        default=[d for d in default_dishes if d in all_fc_dishes],
        placeholder="Busca y selecciona platillos…",
        key="fc_dish",
    )

    # Botón para realizar las petición para hacer las predicciones
    if st.button("Predecir") and sel_fc_dishes:
        mapping = dict(zip(df_ids["platillo_cve"], df_ids["platillo_id"]))
        sel_ids = [mapping[cve] for cve in sel_fc_dishes]
        with st.spinner("Corriendo predicciones…"):
            st.session_state.pred_fc = demand_pred[demand_pred['platillo_id'].isin(sel_ids)]
            st.session_state.pred_ids = sel_fc_dishes
    
    df_results = st.session_state.pred_fc
    if df_results is None:
        st.info("Elige platillos y pulsa **Predecir** para ver resultados.")
        return
    

    # Procesamos el dataset de resultados
    df_results = (
        df_results
        .merge(df_ids, on="platillo_id", how="left")
        .merge(price_df, on="platillo_id", how="left")
    )
    df_results["monto_total"] = df_results["cantidad_pred"] * df_results["precio_unitario"]
    df_results.drop(columns=["precio_unitario", "platillo_id"], inplace=True, errors="ignore")
    df_results["fecha"] = pd.to_datetime(df_results["fecha"])

    # Slider para la ventana de predicciones
    st.markdown("### Ventana de predicciones")
    fc_min, fc_max = df_results["fecha"].min(), df_results["fecha"].max()
    pred_range = st.slider(
        "",
        min_value=fc_min.date(),
        max_value=fc_max.date(),
        value=(fc_min.date(), fc_max.date()),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
        key="pred_slider",
    )

    start_fc, end_fc = map(pd.Timestamp, pred_range)
    fc_filt = df_results[
        df_results["fecha"].between(start_fc, end_fc) &
        df_results["platillo_cve"].isin(sel_fc_dishes)
    ]

    # Métricas de la ventana de predicciones
    # Además de las métricas generales, mostramos un filtro para ver un platillo específico de los predichos
    opciones = ["Todos"] + sel_fc_dishes
    filtro = st.selectbox("Filtrar platillo", opciones, key="table_filter")
    active_dishes = sel_fc_dishes if filtro == "Todos" else [filtro]

    fc_active = fc_filt[fc_filt["platillo_cve"].isin(active_dishes)]
    m1, m2 = st.columns(2)
    m1.metric("Unidades pronosticadas", f"{int(fc_active['cantidad_pred'].sum()):,}")
    m2.metric("Ganancias pronosticadas", f"${fc_active['monto_total'].sum():,.0f} MXN")

    color_seq = px.colors.qualitative.Plotly
    color_map = {dish: color_seq[i % len(color_seq)] for i, dish in enumerate(sel_fc_dishes)}

    # Barplot con los platillos seleccionados
    bar_col, tbl_col = st.columns((3, 2), gap="large")
    with bar_col:
        agg = (
            fc_filt
            .groupby("platillo_cve")["cantidad_pred"]
            .sum()
            .sort_values(ascending=False)
        )
        top_n = min(5, len(agg))
        slice_ = agg.head(top_n)

        fig_bar = go.Figure(
            go.Bar(
                x=slice_.values,
                y=slice_.index.astype(str),
                orientation="h",
                text=[f"{int(v):,d}" for v in slice_.values],
                textposition="auto",
                marker_color=[color_map[d] for d in slice_.index],
            )
        )
        fig_bar.update_yaxes(autorange="reversed")
        fig_bar.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=0, t=25, b=10),
            xaxis_title="Unidades",
            yaxis_title="ID de platillo",
            title="Platillos por Unidades",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla con los resultados de predicciones de ganancias y cantidad predicha
    with tbl_col:
        df_tab = (
            fc_filt[["fecha", "platillo_cve", "cantidad_pred", "monto_total"]]
            .rename(columns={
                "cantidad_pred": "unidades",
                "monto_total": "ganancias"
            })
        )
        df_tab["fecha"] = df_tab["fecha"].dt.date
        if filtro != "Todos":
            df_tab = df_tab[df_tab["platillo_cve"] == filtro]

        header_vals = ["Fecha", "Platillo", "Unidades", "Ganancias"]
        cell_vals = [
            df_tab["fecha"].astype(str),
            df_tab["platillo_cve"],
            df_tab["unidades"],
            df_tab["ganancias"].apply(lambda x: f"${x:,.0f}"),
        ]

        fig_table = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_vals,
                        fill_color="#343A40",
                        font=dict(color="white", size=14),
                        align="center",
                    ),
                    cells=dict(
                        values=cell_vals,
                        fill_color=[["#1E1E1E", "#2A2A2A"]],
                        font=dict(color="white", size=12),
                        align="center",
                        height=30,
                    ),
                    columnwidth=[80, 80, 60, 80],
                )
            ]
        )
        fig_table.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig_table, use_container_width=True)

    # Slider histórico + predicciones
    st.markdown("### Histórico + predicciones")
    hist_min, hist_max = df_demand["fecha"].min(), df_results["fecha"].max()
    hist_range = st.slider(
        "",
        min_value=hist_min.date(),
        max_value=hist_max.date(),
        value=(hist_min.date(), hist_max.date()),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
        key="hist_slider",
    )

    start_h, end_h = map(pd.Timestamp, hist_range)
    hist_line = df_demand[
        df_demand["platillo_cve"].isin(active_dishes) &
        df_demand["fecha"].between(start_h, end_h)
    ]
    fcst_line = df_results[
        df_results["platillo_cve"].isin(active_dishes) &
        df_results["fecha"].between(start_h, end_h)
    ]

    # Line plot 
    fig_line = go.Figure()
    for dish in active_dishes:
        color = color_map[dish]
        h = hist_line[hist_line["platillo_cve"] == dish]
        f = fcst_line[fcst_line["platillo_cve"] == dish]

        fig_line.add_trace(
            go.Scatter(
                x=h["fecha"],
                y=h["cantidad"],
                mode="lines",
                name=f"{dish} (hist)",
                line=dict(color=color, width=2),
            )
        )
        fig_line.add_trace(
            go.Scatter(
                x=f["fecha"],
                y=f["cantidad_pred"],
                mode="lines",
                name=f"{dish} (fcst)",
                line=dict(color=color, dash="dot", width=2),
            )
        )

    fig_line.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Fecha",
        yaxis_title="Unidades",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig_line, use_container_width=True)
