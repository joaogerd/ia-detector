
FROM python:3.10-slim

# Instala dependências básicas
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia os arquivos necessários
COPY requirements.txt ./
COPY src/ ./src
COPY model/ ./model

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar a interface Gradio
CMD ["python", "src/app.py"]
