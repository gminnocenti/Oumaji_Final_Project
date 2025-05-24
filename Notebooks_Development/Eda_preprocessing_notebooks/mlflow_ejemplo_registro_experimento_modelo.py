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
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Infer signature
signature = infer_signature(X_train, clf.predict(X_train))

# Log with MLflow
with mlflow.start_run():    
    # declarar parametros
    mlflow.log_param("random_state", 42)
    # declarar metricas puedes declarar muchas metricas mas
    mlflow.log_metric("accuracy", accuracy)
    # registrar modelo
    mlflow.sklearn.log_model(clf, "model",
                              registered_model_name="MLFLOW_OUMAJI_EJEMPLO",
                              signature=signature)