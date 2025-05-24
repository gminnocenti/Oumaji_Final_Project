#!/usr/bin/env bash
set -euo pipefail

# 1) Load secrets from Key Vault into environment
python3 - <<EOF
import os, sys
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

vault_uri = os.getenv("KEYVAULT_URI")
if not vault_uri:
    sys.exit("ERROR: KEYVAULT_URI not set")

credential = DefaultAzureCredential()
client = SecretClient(vault_url=vault_uri, credential=credential)

for name in [
    "resource-group",
    "storage-account-key",
    "storage-account-name",
    "subscription-id",
    "workspace-name",
    "cluster-name",
]:
    value = client.get_secret(name).value
    os.environ[name.upper().replace("-", "_")] = value
    print(f"✅ Loaded secret: {name}")
EOF

# 2) Exec the command passed to the container (default: CMD)
exec "$@"