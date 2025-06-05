FROM ghcr.io/mlflow/mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host=0.0.0.0", "--backend-store-uri", "file:///ml/mlruns", "--default-artifact-root", "file:///ml/mlartifacts"]
