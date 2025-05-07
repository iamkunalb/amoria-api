FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["uvicorn", "matchmaker_api:app", "--host=0.0.0.0", "--port=8080"]
