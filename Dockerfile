FROM python:3.12.3-slim

WORKDIR /app

COPY . /app

EXPOSE 8501

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD [ "streamlit", "run", "stream.py", "--server.address=0.0.0.0", "--server.port=8501" ]
