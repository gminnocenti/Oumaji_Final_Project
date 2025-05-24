# 1) Install system deps & symlink python3 → python
FROM python:3.11-slim

RUN ln -sf "$(which python3)" /usr/local/bin/python \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy pyproject & README, plus your src tree so that __version__ can be imported
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir .

# 3) Copy the rest (if you have other files) & entrypoint
COPY . /app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 4) Drop privileges, entrypoint, etc.
RUN useradd --create-home kedro && chown -R kedro:kedro /app /entrypoint.sh
USER kedro

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "kedro", "run", "--env=azureml"]