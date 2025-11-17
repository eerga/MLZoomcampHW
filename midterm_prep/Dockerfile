FROM python:3.9-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "./"

ENV PATH="/app/.venv/bin:$PATH"

# installs all of the libraries associated w/ the project
RUN uv sync --locked

COPY "predict.py" "model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]