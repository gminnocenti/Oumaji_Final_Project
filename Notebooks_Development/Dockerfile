FROM ghcr.io/mlflow/mlflow 

EXPOSE 5000 
RUN pip install psycopg2-binary sqlalchemy sqlalchemy-utils
CMD ["mlflow","server","--host=0.0.0.0", "--backend-store-uri", "file:///ml/mlruns","--artifacts-destination","file:///ml/mlartifacts"]