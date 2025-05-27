########################
# Imports
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

########################
# Page configuration
st.set_page_config(
    page_title="Hotel Occupancy Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* tighten the padding around metrics (flash-cards) */
[data-testid="metric-container"] {
    padding: 10px 15px;
}
</style>
""", unsafe_allow_html=True)

########################
# Load data
hist = pd.read_csv("hotel_occupancy_actual.csv", parse_dates=["date"])
fcst = pd.read_csv("hotel_occupancy_forecast.csv", parse_dates=["date"])

# Convenience ranges
full_min, full_max   = hist["date"].min(), fcst["date"].max()
fcst_min, fcst_max   = fcst["date"].min(), fcst["date"].max()

########################
# Sidebar – filters
default_fcst_end   = min(fcst_min + timedelta(days=29), fcst_max)
slider_row = st.columns((1, 1), gap="medium")

with slider_row[0]:
    st.subheader("Historic + Forecast period")
    range_all = st.slider(
        label="",
        min_value=full_min.date(),
        max_value=full_max.date(),
        value=(full_min.date(), full_max.date()),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
        key="range_all",
    )

with slider_row[1]:
    st.subheader("Forecast-only window")
    range_fcst = st.slider(
        label="",
        min_value=fcst_min.date(),
        max_value=fcst_max.date(),
        value=(fcst_min.date(), default_fcst_end.date()),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
        key="range_fcst",
    )

granularity_map = {
    "Daily": "D",          # keep raw data
    "Weekly": "W-MON",     # Monday-anchored weeks
    "Monthly": "M"         # calendar month-end
}

gran_choice = st.radio(
    "Granularity",                 # title shown above the buttons
    options=("Daily", "Weekly", "Monthly"),
    horizontal=True,
    index=0,                       # default = Daily
    key="gran_all",
)


# Convert to Timestamp just once
start_all, end_all   = map(pd.Timestamp, range_all)
start_fcst, end_fcst = map(pd.Timestamp, range_fcst)

########################
# Data subsets
def resample_df(df, rule):          # sums guest counts; switch to .mean() for rates
    if rule == "D":
        return df
    return (
        df.set_index("date")
          .resample(rule)["guests"]
          .sum()
          .reset_index()
    )

# --- base filters (still daily) ---
hist_filt = hist[(hist["date"].between(start_all, end_all))]
fcst_filt = fcst[(fcst["date"].between(start_all, end_all))]
fcst_zoom = fcst[(fcst["date"].between(start_fcst, end_fcst))]

# --- apply one rule to all three ---
rule = granularity_map[gran_choice]
hist_sel  = resample_df(hist_filt,  rule)
fcst_sel  = resample_df(fcst_filt,  rule)
fcst_only = resample_df(fcst_zoom,  rule)

chart_col = st.columns((3, 2), gap="medium")

########################
# Top metrics (flash-cards)
metric_col = st.columns(2)
with metric_col[0]:
    st.metric(
        "Guests (selected period)",
        f"{hist_sel['guests'].sum():,}",
        help="Total number of hotel guests during the chosen historic period.",
    )
with metric_col[1]:
    st.metric(
        "Forecasted guests",
        f"{fcst_only['guests'].sum():,}",
        help="Total forecasted guests in the forecast-only window.",
    )

########################
# Main charts
chart_col = st.columns((3, 2), gap="medium")

# 1) Historic + 30-day forecast in one line chart
with chart_col[0]:
    st.subheader("Historic occupancy with 30-day outlook")

    fig = go.Figure()

    # Historic trace
    fig.add_trace(
        go.Scatter(
            x=hist_sel["date"],
            y=hist_sel["guests"],
            mode="lines",
            name="Actual",
            hovertemplate="%{x|%b %d, %Y}<br>Guests: %{y:,}<extra></extra>",
        )
    )

    # Forecast trace – dotted
    fig.add_trace(
        go.Scatter(
            x=fcst_sel["date"],
            y=fcst_sel["guests"],
            mode="lines",
            name="Forecast",
            line=dict(dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=25, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Guests",
    )
    st.plotly_chart(fig, use_container_width=True)

# 2) Forecast-only chart
with chart_col[1]:
    st.subheader("Forecasted occupancy (zoom)")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=fcst_only["date"],
            y=fcst_only["guests"],
            mode="lines+markers",
            name="Forecast",
            hovertemplate="%{x|%b %d, %Y}<br>Forecast: %{y:,}<extra></extra>",
        )
    )
    fig2.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Date",
        yaxis_title="Guests",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------------------------
# ⬇️ 1. LOAD F&B DATA  (put this near your other read_csv calls)
# ------------------------------------------------------------------------------
fnb_hist = pd.read_csv("fnb_sales_actual.csv",  parse_dates=["date"])
fnb_fcst = pd.read_csv("fnb_sales_forecast.csv", parse_dates=["date"])  # for later tabs

# ------------------------------------------------------------------------------
# ⬇️ 2.  TAB LAYOUT
# ------------------------------------------------------------------------------
tab_fnb_bi, tab_fnb_fc1, tab_fnb_fc2 = st.tabs(
    ["🍽️  F&B BI", "🔮  F&B Forecast I", "🔮  F&B Forecast II"]
)

# ==============================================================================
# TAB 1 - F&B BI
# ==============================================================================
with tab_fnb_bi:

    # 2.1 -- Controls
    fnb_min, fnb_max = fnb_hist["date"].min(), fnb_hist["date"].max()

    ctrl1, ctrl2 = st.columns((3, 1), gap="medium")

    with ctrl1:
        date_range = st.slider(
            "Date range",
            min_value=fnb_min.date(),
            max_value=fnb_max.date(),
            value=(fnb_min.date(), fnb_max.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="fnb_date",
        )

    with ctrl2:
        metric_type = st.radio(
            "Metric",
            options=("Units", "Profit"),
            horizontal=True,
            key="fnb_metric",
        )

    # 2.2 -- Dish selector (full width under the first row)
    all_dishes = sorted(fnb_hist["dish_id"].unique())

    with st.expander("➕ Filter Dish IDs", expanded=False):
        sel_dishes = st.multiselect(
            "Tick one or more dishes",
            options=all_dishes,
            default=all_dishes,                  # everything selected on first load
            placeholder="Type to search…",
            key="fnb_dish",
        )

    # tiny CSS tweak: hide coloured “pill” tags so they don’t sprawl
    st.markdown("""
    <style>
    /* hide the coloured labels that appear after selection */
    span[data-baseweb="tag"] {display: none !important;}
    </style>
    """, unsafe_allow_html=True)

    # 2.3 -- Filter the data
    start_fnb, end_fnb = map(pd.Timestamp, date_range)

    filt = fnb_hist[
        (fnb_hist["date"].between(start_fnb, end_fnb)) &
        (fnb_hist["dish_id"].isin(sel_dishes))
    ]

    # 2.4 -- Metrics
    mcol = st.columns(2)
    mcol[0].metric(
        "Total units sold",
        f"{int(filt['units_sold'].sum()):,}"
    )
    mcol[1].metric(
        "Total profit",
        f"${filt['profit'].sum():,.0f}"
    )

    # 2.5 -- Data prep for charts
    if metric_type == "Units":
        dish_agg = (
            filt.groupby("dish_id")["units_sold"].sum()
            .sort_values(ascending=False)
        )
        weekday_agg = (
            filt.assign(weekday=filt["date"].dt.day_name())
                .groupby("weekday")["units_sold"].sum()
                .reindex(
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                )
        )
        yaxis_title = "Units"
        data_fmt = ",d"
    else:  # Profit
        dish_agg = (
            filt.groupby("dish_id")["profit"].sum()
            .sort_values(ascending=False)
        )
        weekday_agg = (
            filt.assign(weekday=filt["date"].dt.day_name())
                .groupby("weekday")["profit"].sum()
                .reindex(
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                )
        )
        yaxis_title = "Profit ($)"
        data_fmt = ",.0f"

    raw_sizes = [5, 10, 20, 50, len(dish_agg)]
    sizes     = sorted(set(raw_sizes))

    # label each option
    labels         = [("All" if n == len(dish_agg) else str(n)) for n in sizes]
    label_to_value = dict(zip(labels, sizes))

    chosen_label = st.selectbox(
        "Show top …",
        options=labels,
        index=labels.index("All") if "All" in labels else len(labels) - 1,
        key="fnb_top_n",
    )

    top_n = label_to_value[chosen_label]     
    dish_slice = dish_agg.head(top_n)         
    

    # 2.6 -- Charts  (horizontal bar + vertical bar)
    ch_col = st.columns((3, 2), gap="large")

    # -- horizontal bar by dish
    with ch_col[0]:
        fig_dish = go.Figure(
            go.Bar(
                x=dish_slice.values,
                y=dish_slice.index.astype(str),
                orientation="h",
                text=[
                    f"${v:,.0f}" if metric_type == "Profit" else f"{v:,d}"
                    for v in dish_slice.values
                ],
                textposition="auto",
            )
        )
        fig_dish.update_yaxes(autorange="reversed")
        fig_dish.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=0, t=25, b=10),
            xaxis_title="Profit ($)" if metric_type == "Profit" else "Units",
            yaxis_title="Dish ID",
        )
        st.plotly_chart(fig_dish, use_container_width=True)


    # -- vertical bar by weekday
    with ch_col[1]:
        fig_week = go.Figure(go.Bar(
            x=weekday_agg.index,
            y=weekday_agg.values,
            text=[
                f"${v:,.0f}" if metric_type == "Profit" else f"{v:,d}"
                for v in weekday_agg.values
            ],
            textposition="auto",
        ))
        fig_week.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis_title="Weekday",
            yaxis_title=yaxis_title,
        )
        st.plotly_chart(fig_week, use_container_width=True)

# ==============================================================================
# TAB 2 - F&B Forecast I
# ==============================================================================

with tab_fnb_fc1:

    st.subheader("🍽️ 30-Day Dish-level Forecast")

    # 1️⃣ ── controls -----------------------------------------------
    fc_min, fc_max = fnb_fcst["date"].min(), fnb_fcst["date"].max()

    top_row = st.columns((3, 1), gap="medium")

    # -- date slider + Units/Profit toggle --------------------------
    fc_min, fc_max = fnb_fcst["date"].min(), fnb_fcst["date"].max()

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
            options=("Units", "Profit"),
            horizontal=True,
            key="fc_metric",
        )

    # -- dish-ID multiselect (Excel-style filter) -------------------
    with st.expander("➕ Filter Dish IDs", expanded=False):
        all_fc_dishes = sorted(fnb_fcst["dish_id"].unique())
        sel_fc_dishes = st.multiselect(
            "Tick one or more dishes",
            options=all_fc_dishes,
            default=all_fc_dishes,
            placeholder="Type to search…",
            key="fc_dish",
        )

    # hide coloured tags so they don’t sprawl across the page
    st.markdown("""
    <style>span[data-baseweb="tag"] {display:none !important;}</style>
    """, unsafe_allow_html=True)

    # 2️⃣ ── filter df ---------------------------------------------
    # --- filter forecast DF by date & dish ------------------------
    start_fc, end_fc = map(pd.Timestamp, fc_range)
    fc_filt = fnb_fcst[
        (fnb_fcst["date"].between(start_fc, end_fc))
        & (fnb_fcst["dish_id"].isin(sel_fc_dishes))
    ]

    # --- flash cards ----------------------------------------------
    m = st.columns(2)
    m[0].metric("Forecasted units", f"{int(fc_filt['units_sold'].sum()):,}")
    m[1].metric("Forecasted profit", f"${fc_filt['profit'].sum():,.0f}")

    # --- aggregate for bar chart ----------------------------------
    if fc_metric == "Units":
        fc_agg = (
            fc_filt.groupby("dish_id")["units_sold"]
                   .sum()
                   .sort_values(ascending=False)    # ← fixed
        )
        y_title, fmt = "Units", ",d"
    else:
        fc_agg = (
            fc_filt.groupby("dish_id")["profit"]
                   .sum()
                   .sort_values(ascending=False)    # ← fixed
        )
        y_title, fmt = "Profit ($)", ",.0f"

    # ----- Top-N selector (safe) ----------------------------------
    raw_sizes = [5, 10, 20, 50, len(fc_agg)]
    sizes     = sorted(set(raw_sizes))
    labels         = [("All" if n == len(fc_agg) else str(n)) for n in sizes]
    label_to_val   = dict(zip(labels, sizes))
    chosen_label   = st.selectbox(
        "Show top …", labels,
        index=labels.index("All") if "All" in labels else len(labels)-1,
        key="fc_top_n",
    )
    top_n    = label_to_val[chosen_label]
    fc_slice = fc_agg.head(top_n)

    # --- layout (chart + dataframe) -------------------------------
    chart_col, table_col = st.columns((3, 2), gap="large")

    # horizontal bar
    with chart_col:
        fig_fc = go.Figure(go.Bar(
            x=fc_slice.values,
            y=fc_slice.index.astype(str),
            orientation="h",
            text=[f"${v:,.0f}" if fc_metric=="Profit" else f"{v:,d}"
                  for v in fc_slice.values],
            textposition="auto",
        ))
        fig_fc.update_yaxes(autorange="reversed")
        fig_fc.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10,r=0,t=25,b=10),
            xaxis_title=y_title,
            yaxis_title="Dish ID",
            title=f"Top-{top_n if top_n!=len(fc_agg) else 'All'} forecasted dishes",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # dataframe (same style as Tab 1)
    with table_col:
        disp_cols = ["date", "dish_id",
                     "units_sold" if fc_metric=="Units" else "profit"]
        df_disp = fc_filt[disp_cols].sort_values(["date","dish_id"])
        st.dataframe(
            df_disp,
            hide_index=True,
            height=450,
            column_config={
                "units_sold": st.column_config.NumberColumn(format="%d"),
                "profit": st.column_config.NumberColumn(format="$%.0f"),
            },
        )


# ==============================================================================
# TAB 3 - F&B Forecast II
# ==============================================================================

with tab_fnb_fc2:

    st.subheader("📈 Historic vs. Forecast — Dish Lines")

    # ── controls ───────────────────────────────────────────────────
    min_all, max_all = fnb_hist["date"].min(), fnb_fcst["date"].max()

    ctrl_row1 = st.columns((3, 1, 1), gap="medium")
    with ctrl_row1[0]:
        line_range = st.slider(
            "Date range",
            min_value=min_all.date(),
            max_value=max_all.date(),
            value=(min_all.date(), max_all.date()),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
            key="line_date",
        )
    with ctrl_row1[1]:
        line_metric = st.radio(
            "Metric",
            options=("Units", "Profit"),
            horizontal=True,
            key="line_metric",
        )
    with ctrl_row1[2]:
        granularity = st.radio(
            "Granularity",
            options=("Daily", "Weekly"),
            horizontal=True,
            key="line_gran",
        )

    # ── dish selector ------------------------------------------------
    with st.expander("➕ Filter Dish IDs", expanded=False):
        dishes_all = sorted(
            set(fnb_hist["dish_id"]).union(fnb_fcst["dish_id"])
        )
        dishes_sel = st.multiselect(
            "Tick dishes to plot",
            options=dishes_all,
            default=dishes_all[:6],          # show first 6 by default
            placeholder="Type to search…",
            key="line_dish",
        )
    st.markdown(
        "<style>span[data-baseweb='tag']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )

    # ── helper: resample to daily / weekly ---------------------------
    def resample_df(df: pd.DataFrame, rule: str, value_col: str):
        if rule == "D":
            return df
        return (
            df.set_index("date")[value_col]
              .resample(rule)
              .sum()
              .reset_index()
        )

    rule = "D" if granularity == "Daily" else "W-MON"
    val_col = "units_sold" if line_metric == "Units" else "profit"
    y_title = "Units" if line_metric == "Units" else "Profit ($)"
    hover_fmt = ",d" if line_metric == "Units" else ",.0f"

    # ── filter & resample historic + forecast -----------------------
    start_ts, end_ts = map(pd.Timestamp, line_range)

    hist_filt = (
        fnb_hist[fnb_hist["dish_id"].isin(dishes_sel)]
        .loc[lambda d: d["date"].between(start_ts, end_ts)]
    )
    fcst_filt = (
        fnb_fcst[fnb_fcst["dish_id"].isin(dishes_sel)]
        .loc[lambda d: d["date"].between(start_ts, end_ts)]
    )

    # resample
    hist_rs = resample_df(hist_filt, rule, val_col)
    fcst_rs = resample_df(fcst_filt, rule, val_col)

    # ── build line traces ------------------------------------------
    fig = go.Figure()

    for dish in dishes_sel:
        # historic trace
        hd = hist_rs[hist_rs["dish_id"] == dish] if "dish_id" in hist_rs else \
             hist_rs.assign(dish_id=hist_filt["dish_id"].iloc[0])
        fd = fcst_rs[fcst_rs["dish_id"] == dish] if "dish_id" in fcst_rs else \
             fcst_rs.assign(dish_id=fcst_filt["dish_id"].iloc[0])

        fig.add_trace(
            go.Scatter(
                x=hd["date"],
                y=hd[val_col],
                mode="lines",
                name=f"{dish} (hist)",
                hovertemplate=f"{dish}<br>%{{x|%Y-%m-%d}}<br>%{{y:{hover_fmt}}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fd["date"],
                y=fd[val_col],
                mode="lines",
                name=f"{dish} (fcst)",
                line=dict(dash="dot"),
                hovertemplate=f"{dish}<br>%{{x|%Y-%m-%d}}<br>%{{y:{hover_fmt}}}<extra></extra>",
            )
        )

    # ── figure layout ----------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Date",
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)



########################
# About / footnote
with st.expander("ℹ️ About this dashboard", expanded=False):
    st.markdown(
        """
* **Data source**: internal PMS exports for actuals, plus your preferred time-series model for forecasts.  
* **Flash-cards**: summed guests over the selected ranges (you can switch to *average occupancy rate* if that’s more meaningful).  
* **Filters** act instantly; no “apply” button needed thanks to Pandas boolean indexing.  
* **Visuals**: Plotly Dark template for quick PowerBI-like polish; dotted line = future.  
"""
    )