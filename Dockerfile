FROM nvidia/cuda:12.6.2-base-ubuntu20.04

RUN apt-get update && apt-get install -y \ 
    python3 \
    python3-pip \
    libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /

RUN pip3 install --no-cache-dir -r requirements.txt

RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]

ENV NLTK_DATA=/usr/local/nltk_data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]