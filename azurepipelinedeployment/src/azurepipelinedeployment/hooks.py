from kedro.framework.hooks import hook_impl
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

class VaultSecretsHook:
    @hook_impl
    def before_node_run(self):
        vault_uri = os.getenv("KEYVAULT_URI")
        if not vault_uri:
            raise RuntimeError("❌ KEYVAULT_URI no está definida.")

        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_uri, credential=credential)

        secrets_to_load = [
            "resource-group",
            "storage-account-key",
            "storage-account-name",
            "subscription-id",
            "workspace-name",
            "cluster-name",
        ]

        for name in secrets_to_load:
            try:
                value = client.get_secret(name).value
                os.environ[name.upper().replace("-", "_")] = value
                print(f"✅ Se cargó el secreto: {name}")
            except Exception as e:
                raise RuntimeError(f"❌ No se pudo obtener el secreto '{name}': {e}")