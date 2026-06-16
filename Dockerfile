FROM python:3.11-slim
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY data.py .
COPY graph.py .
COPY app.py .
COPY crop.db .
 
# Build crop.db from generated fake data at image build time
 
ENV PORT=8000
ENV DB_PATH=/app/crop.db
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.2:3b-instruct-fp16
 
EXPOSE 8000
 
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "--capture-output", "--log-level", "debug"]
