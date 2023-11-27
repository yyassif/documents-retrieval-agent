FROM python:3.9-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3333

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3333"]