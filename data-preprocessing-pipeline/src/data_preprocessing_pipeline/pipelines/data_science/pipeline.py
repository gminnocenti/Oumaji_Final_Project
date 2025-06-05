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
        outputs = ["X_train", "X_test", "y_train", "y_test"],
        name = "train_test_split_occupancy",
        ),
        node(
            func = sarimax,
            inputs = ["X_train", "y_train", "X_test", "y_test"],
            outputs = "results_occupancy",
            name = "sarimax_model"
        ),
        node(
            func = train_test_split_demand,
            inputs = "daily_demand",
            outputs = ["X_train_demand", "X_test_demand", "y_train_demand", "y_test_demand", "horizon"],
            name = "train_test_split_demand"
        ),
        node(
            func = lightgbm_pca,
            inputs = ["X_train_demand", "y_train_demand", "X_test_demand", "y_test_demand", "results_occupancy", "horizon","dishes_mapping"],
            outputs = "results_demand",
            name = "lightgbm_model"
        )



    ])


