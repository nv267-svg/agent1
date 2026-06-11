FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync

COPY . .

EXPOSE 8080

CMD [".venv/bin/python", "main.py"]
