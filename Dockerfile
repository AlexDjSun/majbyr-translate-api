FROM alexthesun/cuda116-majbyrapi-base:v0.6

COPY . /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]