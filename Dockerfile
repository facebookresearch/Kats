FROM python:3.7
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt