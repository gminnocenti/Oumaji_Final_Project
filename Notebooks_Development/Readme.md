En este folder se deben desarrollar desarrollar todos los notebooks de prueba para los datasets y modelos.

# Instrucciones para desplegar contenedor de MLFLOW

1. En una terminal moverme al directorio del folder `Notebooks_Deployment`
```
cd Notebooks_Development
```
corro este archivo
```
python fix_meta.py
```
2. En el directorio del folder corro en la terminal
Para construir la imagen

```
docker compose up --build

```
3. Correr contenedor con la imagen ya creada
si ya tengo la imagen puedo correr directamente el contenedor desde Docker Desktop el contenedor se llama `mlflow_Oumaji_experiment` o me muevo en la terminal  al folder `Notebooks_Development` y corro


```
docker compose start
```

**Si ya corres el paso 2 una vez puedes correr la imagen desde docker desktop**

## Como configurar tu experimento y registrar un modelo.

Es muy importante que cuando crees un modelo sigas los siguientes pasos. 
En mlflow puedes declarar experimentos y modelos. Puedes declarar todos los modelos que quieres desde un experimento.Cada vez que yo corro y registro un modelo bajo un experimento de mlflow se guardan varios archivos relevantes al modelo. Como el pickle file y diferentes metricas, o parametros del modelo. Estos se guardan bajo las carpetas de mlruns y mlartifacts. Es **importante que se mantenga el nombre del experimento** en todos los notebooks para que todos los modelos sean guardados bajo la misma carpeta de mlruns y mlartifacts.
### Como incializar un experimento y registrar un modelo de mlflow

1. conectarse al servidor del contenedor y conectarse al experimento.
**IMPORTANTE**
Tu al correr el contenedor estas inicializando un servidor local de mlflow el cual si te metes a tu browser con la siguiente direccion lo puedes encontrar: `http://localhost:5001/` . Ahi puedes ver el experimento y todos los modelos corridos.
**Como declarar el experimento y conectarse a este servidor desde tu notebook**
en tu notebook debes poner esto en la primera celda la direccion del servidor  `http://localhost:5001/` y el nombre del experimento `Oumaji_Model_Development`

```python
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set tracking URI and experiment
# me conecto al servidor del contenedor
mlflow.set_tracking_uri("http://localhost:5001")
# me conecto al experimento del proyecto
mlflow.set_experiment("Oumaji_Model_Development")
```

2. Ejemplo de desarrollo de modelo y registrarlo

```python 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Oumaji_Model_Development")

# Load dataset
data = load_wine()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
# declaro metrica
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Infer signature
signature = infer_signature(X_train, clf.predict(X_train))

## REGiSTRO DEL MODELO
with mlflow.start_run():
    mlflow.log_param("random_state", 42)
    # aca podemos agregar las metricas que queramos , MAE F1, etc
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model",
                              registered_model_name="MLFLOW_OUMAJI_EJEMPLO",
                              signature=signature)
```

todo este codigo lo puedes ver completo lo puedes encontrar en el archivo `Notebooks_Development/mlflow_ejemplo_registro_experimento_modelo.py`
## Metodologia de registro de modelo 
Caso Hipotetico
Axl y yo estamos tratando de ver que modelo es mejor. El va a usar un arima y yo un sarima pero queremos ver el desempeño de su modelo y de el mio bajo el mismo experimento. Estaremos desarrollando notebooks diferentes pero la primera celda de ambos se debe ver asi :
```
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set tracking URI and experiment
# me conecto al servidor del contenedor
mlflow.set_tracking_uri("http://localhost:5001")
# me conecto al experimento del proyecto
mlflow.set_experiment("Oumaji_Model_Development")
```
Despues cada quien hace el preprocessing necesario y el modelo. A la hora de registar el modelo se debe ver muy similar lo unico que debe ser diferente es el `registered_model_name`. 
Axl debe ponerle en `registered_model_name` = "Arima-model" y yo debo ponerle `registered_model_name` = "Sarima-model".
```
with mlflow.start_run():
    mlflow.log_param("random_state", 42)
    # aca podemos agregar las metricas que queramos , MAE F1, etc
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model",
                              registered_model_name="<NOMBRE-MODELO>",
                              signature=signature)
```

De esta manera el puede cambiar parametros de su modelo y correrlo n veces. Todas esas versionas seran registradas en las carpetas de `mlrun` y `mlartifacts` y las mias tambien. Cuando el termina hace push a su trabajo. Yo hago un git pull y tendre todos los archivos y metricas de sus modelos en mi proyecto. Y los puedo visualizar en la direccion  `http://localhost:5001` siempre y cuando haya corrido mi contenedor antes.

# Resumen 
1. construir imagen y correr contenedor

una vez ya tengo mi contenedor corriendo puedo crear mi propio modelo.

2. Creo un nuevo jupyter notebook donde voy a desarrollar un modelo.

3. me conecto al servidor del contenedor `http://localhost:5001/` y al experimento del proyecto `Oumaji_Model_Development` desde mi notebook.
```
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Oumaji_Model_Development")
```
4. desarrollo mi modelo y lo registro de la siguiente forma
**IMPORTANTE**
Siempre incluir metricas, el model signature y declarar un Nombre relevante al tipo del modelo que estas desarrollando
```
with mlflow.start_run():
    mlflow.log_param("random_state", 42)
    # aca podemos agregar las metricas que queramos , MAE F1, etc
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model",
                              registered_model_name="<NOMBRE-MODELO>",
                              signature=signature)
```

tus resultados seran guardados en las carpetas de mlruns y mlartifacts.