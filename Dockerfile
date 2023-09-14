FROM python:3.7-slim
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y libgomp1
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt