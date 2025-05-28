"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import sarimax, lightgbm_pca


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func = sarimax,
            inputs = "daily_occupancy",
            outputs = None,
            name = "sarimax_model"
        ),
        node(
            func = lightgbm_pca,
            inputs = ["daily_demand","feature_cols"],
            outputs = "mae_per_plt",
            name = "lightgbm_model"
        )



    ])


