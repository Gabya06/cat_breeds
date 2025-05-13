# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies first to leverage caching
RUN pip install --upgrade pip && \
    pip install torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY ./src /app/src
COPY ./db /app/db
COPY ./data /app/data

RUN pip install --no-cache-dir -e .

# Set the working directory to app/src
WORKDIR /app/src

# Expose ports
# Local development Streamlit default
EXPOSE 8501  
# Cloud Run expected port
EXPOSE 8080  

# Run the app
# CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false

