"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_reservations, generate_daily_occupancy, clean_dishes, generate_daily_demand


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = clean_reservations,
            inputs=dict(df="iar_reservaciones",numeric_columns_reservations="params:numeric_columns_reservations"),
            outputs = "data_cleaned_reservations",
            name = "clean_reservations",
        ),
        node(
            func = generate_daily_occupancy,
            inputs = "data_cleaned_reservations",
            outputs = "daily_occupancy",
            name = "generate_daily_occupancy",
        ),
        node(
            func = clean_dishes,
            inputs = dict(df = "iaab_Detalles_Vtas", numeric_columns_dishes = "params:numeric_columns_dishes"),
            outputs = "data_cleaned_dishes",
            name = "cean_dishes",
        ),
        node(
            func = generate_daily_demand,
            inputs = dict(df="data_cleaned_dishes", df_occupancy="daily_occupancy"),
            outputs =  ["daily_demand", "dishes_mapping"],
            name = "generate_weekly_demand",
        )

    ])
