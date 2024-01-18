FROM alexthesun/cuda116-majbyrapi-base:v0.5

COPY . /

RUN pip3 install --no-cache-dir nltk

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]