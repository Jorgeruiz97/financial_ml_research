FROM python:3.7-slim-buster
# Install python packages
RUN pip install mlflow pymysql

EXPOSE 5000

CMD mlflow server \
  --backend-store-uri /mnt/persistent-disk \
  --default-artifact-root s3://my-mlflow-bucket/ \
  --host 0.0.0.0
