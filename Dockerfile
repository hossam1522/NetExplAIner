FROM debian:stable-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates make python3 && \ 
    rm -rf /var/lib/apt/lists/* && \
    useradd -m test
    
ENV PATH="/home/test/.local/bin:$PATH"
ENV UV_CACHE_DIR=/home/test/.cache/uv
ENV UV_PROJECT_ENVIRONMENT=/home/test/.venv

RUN mkdir -p ${UV_CACHE_DIR} ${UV_PROJECT_ENVIRONMENT} && \
    chmod -R a+w ${UV_CACHE_DIR} ${UV_PROJECT_ENVIRONMENT}

USER test

WORKDIR /app/test

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENTRYPOINT ["make", "test"]
