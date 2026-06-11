FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv sync

EXPOSE 8080

CMD ["uv", "run", "python", "main.py"]
