FROM python:3.12-slim
WORKDIR /app
# This line is the secret: it installs the tools permanently into the container
RUN pip install flask pandas
COPY . .
CMD ["python", "main.py"]
