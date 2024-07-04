FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

COPY src/model .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]