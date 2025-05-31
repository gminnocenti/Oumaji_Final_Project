# 1) Imagen base y versión de Python
FROM python:3.11-slim

# 2) Asegurarnos de que pip, setuptools y build estén actualizados
ENV PYTHONUNBUFFERED=1
RUN python -m pip install --upgrade pip setuptools build

# 3) Instalar dependencias de sistema necesarias para compilar
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4) Establecer directorio de trabajo (el contexto dentro del contenedor)
WORKDIR /app

# 5) Copiar únicamente los archivos de metadatos (para aprovechar la cache de Docker)
COPY pyproject.toml README.md /app/

# 6) Instalar todas las dependencias definidas en pyproject.toml
#    Esto asume que pyproject.toml tiene la sección [project] con 'dependencies'
RUN pip install .

# 7) Copiar el resto del código (source) dentro del contenedor
COPY src/ /app/src/
COPY page_icon.webp /app/

# 8) Exponer el puerto en el que correrá Streamlit 
EXPOSE 8501

# 9) Comando por defecto para arrancar la aplicación
#    Streamlit ejecuta src/main.py en la dirección 0.0.0.0:8501
CMD ["streamlit", "run", "src/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
