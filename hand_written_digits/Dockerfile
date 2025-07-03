FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY demo_app.py .
COPY handwritten.keras .
COPY templates/ templates/
COPY static/ static/
RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=demo_app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5002
EXPOSE 5002

CMD ["flask", "run"]
