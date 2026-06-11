FROM python:3.12-slim
WORKDIR /app
RUN pip install flask pandas
COPY . .
CMD ["python", "main.py"]
