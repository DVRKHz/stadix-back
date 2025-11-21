# imagen ligera de Python
FROM python:3.10-slim

# Evita que Python genere archivos .pyc y fuerza salida a consola
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Directorio de trabajo dentro del contenedor
WORKDIR /code

# primero los requerimientos (para aprovechar caché de Docker)
COPY requirements.txt /code/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# el resto del código
COPY . /code/

# Comando para correr la app con Hot-Reloading de --realod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]