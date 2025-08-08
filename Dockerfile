FROM python:3.12.3-slim

WORKDIR /app

COPY . /app/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

CMD [ "streamlit", "run", "stream.py" ]


3,78,50,32,88,31,0.248,26,1