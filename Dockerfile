FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    APP_HOME="/home/app"

ENV PATH="$POETRY_HOME/bin:$APP_HOME/.venv/bin:$PATH"
ENV PYTHONPATH=$APP_HOME
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
        vim \
        tmux \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock* ./


FROM base AS dev

ARG INSTALL_DEV=true

RUN poetry config virtualenvs.create false && \
    if [ "$INSTALL_DEV" = "true" ]; then \
        poetry install --no-root; \
    else \
        poetry install --no-root --only main; \
    fi && \
    rm -rf ~/.cache/pypoetry/{cache, artifacts}

COPY . .


FROM base AS ci

RUN poetry config virtualenvs.create false && \
    poetry install --no-root && \
    rm -rf ~/.cache/pypoetry/{cache, artifacts}

COPY mcts_llm mcts_llm
COPY tests tests
COPY scripts scripts
