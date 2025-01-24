FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt requirements_frontend.txt
COPY src src/

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

EXPOSE 8080
EXPOSE 8081

ENTRYPOINT ["sh", "-c", "streamlit run src/fruit_vegetable_classification/frontend.py --server.port=8081 --server.address=0.0.0.0"]