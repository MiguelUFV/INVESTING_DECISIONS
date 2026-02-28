# Aura Wealth OS - Dockerfile de Produccion (Nube)
# Recomendado para despliegue en Render, Railway, AWS o GCP.

# Usa el motor nativo de Python ultraligero
FROM python:3.10-slim

# Evita que Python genere archivos .pyc (.pyo)
ENV PYTHONDONTWRITEBYTECODE 1
# Fuerza a que el output de consola de Python se mande directo al terminal local
ENV PYTHONUNBUFFERED 1

# Crea y salta al directorio de infraestructura
WORKDIR /app

# Descarga dependencias a nivel de Sistema Operativo para computacion cuantica
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copia binarios de dependencia y pre-instala. Cero cache para menor peso del build
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Inyecta toda nuestra suite de codigos financieros al contenedor asilado
COPY . .

# Expone el puerto por el que Streamlit emite senales HTTP
EXPOSE 8501

# Define el chequeo de disponibilidad en produccion
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Al arrancar, enciende el Quantitative Engine en modo host 0.0.0.0 absoluto
CMD ["streamlit", "run", "dashboard_financiero.py", "--server.port=8501", "--server.address=0.0.0.0"]
