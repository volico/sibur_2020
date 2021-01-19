FROM python:3.7

COPY docker_predictions.py server.py
COPY requirements.txt .

EXPOSE 5000

RUN pip install --no-cache-dir -r requirements.txt

CMD python server.py