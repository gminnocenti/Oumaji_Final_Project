"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd
import holidays

def clean_reservations(df: pd.DataFrame, numeric_columns_reservations: list[str]) -> pd.DataFrame:
    """
    Removes rows where:
    - 'ID_estatus_reservaciones' is between 0 and 6 (inclusive), basically
    meaning that that reservation did not arrive, was not completed among other things.
    - Any numeric column has an outlier (based on IQR).

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list[str]): List of numeric columns to evaluate.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    df_clean = df_clean[~df_clean["ID_estatus_reservaciones"].between(0, 6)]

    outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)

    for col in numeric_columns_reservations:
        if df_clean[col].dropna().nunique() < 2:
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        is_outlier = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_mask = outlier_mask | is_outlier

    return df_clean[~outlier_mask]

def generate_daily_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a daily occupancy DataFrame by expanding date ranges between check-in and check-out,
    only if 'h_num_noc' (number of nights) matches the date difference.

    Then it filters the DataFrame to include only dates up to '2020-03-13', which is the 
    start date of the COVID-19 pandemic in Mexico, meaning that reservations dramatically decreased.

    It also creates some time features:
        - 'dia_festivo': 1 if the date is a holiday, 0 otherwise.
        - 'lag_1': occupancy of the previous day.
        - 'lag_2': occupancy of two days before.
        - 'lag_4': occupancy of four days before.

    Args:
        df (pd.DataFrame): Input DataFrame with 'h_fec_lld', 'h_fec_sda', 'h_num_per', 'h_num_noc'.

    Returns:
        pd.DataFrame: A DataFrame with columns 'fecha', 'ocupacion', 'dia_festivo', 'lag_1', 'lag_2', 'lag_4'.
    """
    df = df.dropna(subset=["h_fec_lld", "h_fec_sda", "h_num_per", "h_num_noc"])

    df["h_fec_lld"] = pd.to_datetime(df["h_fec_lld"], format="%Y%m%d", errors="coerce")
    df["h_fec_sda"] = pd.to_datetime(df["h_fec_sda"], format="%Y%m%d", errors="coerce")
    df["h_num_noc"] = pd.to_numeric(df["h_num_noc"], errors="coerce")
    df = df[df["h_num_noc"] == (df["h_fec_sda"] - df["h_fec_lld"]).dt.days]

    records = []
    for _, row in df.iterrows():
        for date in pd.date_range(row["h_fec_lld"], row["h_fec_sda"] - pd.Timedelta(days=1)):
            records.append({
                "fecha": date,
                "ocupacion": row["h_num_per"]
                })

    occ_df = pd.DataFrame(records)
    occ_df = occ_df.groupby("fecha", as_index=False).sum()
    occ_df = occ_df[occ_df['fecha'] <= '2020-03-13']

    years = range(occ_df["fecha"].dt.year.min(), occ_df["fecha"].dt.year.max() + 1)
    mx_holidays = holidays.MX(years=years)
    occ_df["dia_festivo"] = occ_df["fecha"].isin(mx_holidays).astype(int)

    occ_df = occ_df.sort_values("fecha")
    for lag in [1, 2, 4]:
        occ_df[f"lag_{lag}"] = occ_df["ocupacion"].shift(lag)

    occ_df = occ_df.fillna(0)

    return occ_df

def clean_dishes(df: pd.DataFrame, numeric_columns_dishes: list[str]) -> pd.DataFrame:

    """
    Removes rows where:
    - Any numeric column has an outlier (based on IQR).

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list[str]): List of numeric columns to evaluate.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)

    for col in numeric_columns_dishes:
        if df_clean[col].dropna().nunique() < 2:
            continue

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        is_outlier = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_mask = outlier_mask | is_outlier

    return df_clean[~outlier_mask]

def generate_daily_demand(df: pd.DataFrame, df_occupancy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a daily demand DataFrame.
    It also creates a mapping DataFrame for platillo_cve to platillo_id.
    The function filters the data to include only dates in the daily_occupancy dataset.
    It also creates some time features:
        - 'lag_1': demand of the previous days.
        - 'lag_7': demand one week ago.
        - 'dia_festivo': 1 if the date is a holiday, 0 otherwise.
        - 'dia_semana': 0 if its monday, 1 if its tuesday, etc.

    Also, it filters dishes with low revenue, keeping only the top 80% of revenue-generating dishes.
    Thus helping to reduce the dimensionality of the data and focus on the most important dishes.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'fec_com', 'platillo_cve', 'monto_total'.
        df_occupancy (pd.DataFrame): DataFrame with daily occupancy data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - daily_df: A DataFrame with weekly demand data.
            - mapping_df: A DataFrame with mapping of platillo_cve to platillo_id.

    """

    df["fec_com"] = pd.to_datetime(df["fec_com"], format="%Y%m%d", errors="coerce")
    df_occupancy["fecha"] = pd.to_datetime(df_occupancy["fecha"], errors="coerce")

    valid_dates = df_occupancy["fecha"].unique()
    df = df[df["fec_com"].isin(valid_dates)]

    grouped = df.groupby(["fec_com", "platillo_cve"]).agg(
    cantidad=('platillo_cve', 'size'),
    monto_total=('monto_total', 'sum')
            ).reset_index().rename(columns={"fec_com": "fecha"})

    all_dates = pd.date_range(grouped["fecha"].min(), grouped["fecha"].max())
    all_combinations = pd.MultiIndex.from_product([all_dates, grouped["platillo_cve"].unique()], names=["fecha", "platillo_cve"])
    daily_df = grouped.set_index(["fecha", "platillo_cve"]).reindex(all_combinations, fill_value=0).reset_index()
    daily_df = daily_df[daily_df['fecha'] <= '2020-03-13']

    platillo_id_map = {cve: idx for idx, cve in enumerate(sorted(daily_df["platillo_cve"].unique()))}
    daily_df["platillo_id"] = daily_df["platillo_cve"].map(platillo_id_map)

    daily_df = daily_df.merge(df_occupancy[["fecha", "ocupacion"]], on="fecha", how="left")

    years = range(daily_df["fecha"].dt.year.min(), daily_df["fecha"].dt.year.max() + 1)
    mx_holidays = holidays.MX(years=years)

    daily_df["dia_festivo"] = daily_df["fecha"].isin(mx_holidays).astype(int)
    daily_df["dia_semana"] = daily_df["fecha"].dt.weekday
    daily_df = daily_df.sort_values(by=["platillo_id", "fecha"])
    daily_df["lag_1"] = daily_df.groupby("platillo_id")["cantidad"].shift(1)
    daily_df["lag_7"] = daily_df.groupby("platillo_id")["cantidad"].shift(7)

    revenue_por_platillo = daily_df.groupby("platillo_id")["monto_total"].sum().reset_index()

    revenue_por_platillo = revenue_por_platillo.sort_values("monto_total", ascending=False)
    revenue_por_platillo["revenue_acum"] = revenue_por_platillo["monto_total"].cumsum()
    revenue_por_platillo["revenue_ratio"] = revenue_por_platillo["revenue_acum"] / revenue_por_platillo["monto_total"].sum()

    platillos_top_80 = revenue_por_platillo[revenue_por_platillo["revenue_ratio"] <= 0.8]["platillo_id"]
    daily_df = daily_df[daily_df["platillo_id"].isin(platillos_top_80)]

    mapping_df = pd.DataFrame(list(platillo_id_map.items()), columns=["platillo_cve", "platillo_id"])
    mapping_df = mapping_df[mapping_df["platillo_id"].isin(platillos_top_80)].reset_index(drop=True)


    daily_df = daily_df.fillna(0)
    daily_df.drop('platillo_cve', axis=1, inplace=True)

    return daily_df, mapping_df












