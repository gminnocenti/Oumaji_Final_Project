"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_test_split_occupancy, sarimax


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
        func = train_test_split_occupancy,
        inputs = "daily_occupancy",
        outputs = ["X_train", "X_test", "y_train", "y_test"],
        name = "train_test_split_occupancy_node",
        ),
        node(
            func = sarimax,
            inputs = ["X_train", "y_train", "X_test", "y_test"],
            outputs = "results_occupancy",
            name = "sarimax_model"
        )



    ])


