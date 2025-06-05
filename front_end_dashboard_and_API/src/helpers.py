"""Helper functions for main logic."""

import pandas as pd


def resample_df(df: pd.DataFrame, rule):
    """Sums guest counts in the DataFrame based on the specified resampling rule."""
    if rule == "D":
        return df
    return df.set_index("fecha").resample(rule)["ocupacion"].sum().reset_index()


def resample_df_column(df: pd.DataFrame, rule: str, value_col: str):
    if rule == "D":
        return df
    return df.set_index("fecha")[value_col].resample(rule).sum().reset_index()
