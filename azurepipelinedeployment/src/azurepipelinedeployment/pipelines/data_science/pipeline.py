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
            outputs = ["occupancy_predictions","list_last_date"],
            name = "sarimax_model"
        ),
        node(
            func = lightgbm_pca,
            inputs = ["daily_demand","feature_cols","list_last_date","emb_df","k_list","occupancy_predictions"],
            outputs = ["mae_per_plt","df_pred_demand"],
            name = "lightgbm_model"
        )



    ])


