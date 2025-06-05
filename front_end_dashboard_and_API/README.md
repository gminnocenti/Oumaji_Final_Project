# crear las variables de entorno de mis blob storage, 

1. **Subir los secretos necesarios para conectarme a mi blob storage, y a cada model endpoint.**
vas a necesitar subir a tu key chain en azure los siguientes secretos:
- `Blob Storage`:
    - Storage Account Name
    - Storage Accont Key
- Para cada modelo debo incluir:
    - endpoint_Url
    - model api key

2. **subir los secretos necesarios a mi key vault** si no los tengo 
```
az login
```
```
az keyvault secret set --vault-name <your-key-vault-name> --name model-endpoint-url        --value "<your-model-endpoint-url>"
az keyvault secret set --vault-name <your-key-vault-name> --name model-api-key        --value "<your-model-api-key>"
az keyvault secret set --vault-name <your-key-vault-name> --name storage-account-name  --value "<your-storage-account-name>"
az keyvault secret set --vault-name <your-key-vault-name> --name storage-account-key   --value "<your-storage-account-key>"
```


3. **Construir imagen y pushear imagen a acr**
```
az login
az acr login \
  --name <acr-name> \
  --resource-group <resource-group-name>
```

```
docker buildx build \
  --platform linux/amd64 \
  --no-cache \
  --tag <nombre-acr>.azurecr.io/<nombre-imagen>:latest \
  --push .
```

# Crear Recursos necesarios
1. **login**
```
az login
```
2. **registrar Microsoft.Web a tu subscripcion**
```
az provider register --namespace Microsoft.Web

```
```
az provider show --namespace Microsoft.Web --query "registrationState"

```
3. **crear un App service plan**
dentro del app service plan puedes tener muchas aplicaciones lo puedes ver como un storage account donde puedes guardar todas tus aplicaciones
```
az appservice plan create \
  --name <nombre-que-quieres-para-tu-servicio> \
  --resource-group <resource-group-name> \
  --sku B1 \
  --is-linux

```

4. **Crear un web app dentro de mi App service plan**
```
az webapp create \
  --resource-group <resource-group-name> \
  --plan <app-service-plan-name> \
  --name <name-you-want-for-your-app> \
  --deployment-container-image-name <nombre-acr>.azurecr.io/<nombre-imagen>:latest

```

5. **darle permiso a mi web app de `arcpull` a mi azure container registry para eso debo encontrar el principal id manage identity de mi web app.**
```
PRINCIPAL_ID=$(az webapp identity assign \
  --name <nombre-de-tu-web-app> \
  --resource-group <resource-group-name> \
  --query principalId -o tsv)
```
```
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<resource-group-name>/providers/Microsoft.ContainerRegistry/registries/<ACR_NAME> \
  --role "AcrPull"
```
Me  voy a mi container registry y le doy a mi web app el role de arc pull desde azure ui
![alt text](image.png)
6. **darle acceso a mi web app de mi key vault usando el principal id de mi web app**
```
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Key Vault Secrets User" \
  --scope $(az keyvault show \                                               
              --name <nombre-key-vault> \
              --resource-group <nombre-resource-group> \
              --query id -o tsv)
```

7. **agregar env variable a mi webb app**

```
az webapp config appsettings set \
  --resource-group <resource-group-name> \
  --name <web-app-name> \
  --settings KEYVAULT_URI="https://<nombre-key-vault>.vault.azure.net/"

```
8. **me voy a azure ui a mi web app / deployment center selecciono authentification managed identity y selecciono el registry y imagen que quiero**
![alt text](dash.png)
y le das save
9. restart webb application
```
az webapp restart \
  --resource-group <nombre-resource-group> \
  --name <nombre-streamlit-application>

```

10. **en el siguiente link puedes ver tu aplicacion**
`https://<nombre-aplicacion>.azurewebsites.net`


