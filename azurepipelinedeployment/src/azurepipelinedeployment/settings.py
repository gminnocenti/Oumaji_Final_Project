import os
from omegaconf import OmegaConf

# Register ENV_VAR resolver to pull from environment variables
OmegaConf.register_resolver(
    "ENV_VAR",
    lambda var_name, default=None: os.environ.get(var_name, default)
)

                              
from azurepipelinedeployment.hooks import VaultSecretsHook              

HOOKS = (VaultSecretsHook(),)
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-mlflow", "kedro_mlflow")


# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog