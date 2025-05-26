"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_test_split_occupancy, sarimax, train_test_split_demand, lightgbm_pca


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
        func = train_test_split_occupancy,
        inputs = "daily_occupancy",
        outputs = ["X","y"],
        name = "train_test_split_occupancy",
        ),
        node(
            func = sarimax,
            inputs = ["X","y"],
            outputs = None,
            name = "sarimax_model"
        ),
        node(
            func = train_test_split_demand,
            inputs = "daily_demand",
            outputs = ["X_demand", "y_demand"],
            name = "train_test_split_demand"
        ),
        node(
            func = lightgbm_pca,
            inputs = ["X_demand", "y_demand","dishes_mapping"],
            outputs = ["mae_per_plt"],
            name = "lightgbm_model"
        )



    ])


